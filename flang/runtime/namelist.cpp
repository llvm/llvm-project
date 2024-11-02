//===-- runtime/namelist.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "namelist.h"
#include "descriptor-io.h"
#include "emit-encoded.h"
#include "io-stmt.h"
#include "flang/Runtime/io-api.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace Fortran::runtime::io {

// Max size of a group, symbol or component identifier that can appear in
// NAMELIST input, plus a byte for NUL termination.
static constexpr std::size_t nameBufferSize{201};

static inline char32_t GetComma(IoStatementState &io) {
  return io.mutableModes().editingFlags & decimalComma ? char32_t{';'}
                                                       : char32_t{','};
}

bool IONAME(OutputNamelist)(Cookie cookie, const NamelistGroup &group) {
  IoStatementState &io{*cookie};
  io.CheckFormattedStmtType<Direction::Output>("OutputNamelist");
  io.mutableModes().inNamelist = true;
  char comma{static_cast<char>(GetComma(io))};
  ConnectionState &connection{io.GetConnectionState()};
  // Internal functions to advance records and convert case
  const auto EmitWithAdvance{[&](char ch) -> bool {
    return (!connection.NeedAdvance(1) || io.AdvanceRecord()) &&
        EmitAscii(io, &ch, 1);
  }};
  const auto EmitUpperCase{[&](const char *str) -> bool {
    if (connection.NeedAdvance(std::strlen(str)) &&
        !(io.AdvanceRecord() && EmitAscii(io, " ", 1))) {
      return false;
    }
    for (; *str; ++str) {
      char up{*str >= 'a' && *str <= 'z' ? static_cast<char>(*str - 'a' + 'A')
                                         : *str};
      if (!EmitAscii(io, &up, 1)) {
        return false;
      }
    }
    return true;
  }};
  // &GROUP
  if (!(EmitWithAdvance('&') && EmitUpperCase(group.groupName))) {
    return false;
  }
  auto *listOutput{io.get_if<ListDirectedStatementState<Direction::Output>>()};
  for (std::size_t j{0}; j < group.items; ++j) {
    // [,]ITEM=...
    const NamelistGroup::Item &item{group.item[j]};
    if (listOutput) {
      listOutput->set_lastWasUndelimitedCharacter(false);
    }
    if (!(EmitWithAdvance(j == 0 ? ' ' : comma) && EmitUpperCase(item.name) &&
            EmitWithAdvance('=') &&
            descr::DescriptorIO<Direction::Output>(io, item.descriptor))) {
      return false;
    }
  }
  // terminal /
  return EmitWithAdvance('/');
}

static constexpr bool IsLegalIdStart(char32_t ch) {
  return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '_' ||
      ch == '@' || ch == '$';
}

static constexpr bool IsLegalIdChar(char32_t ch) {
  return IsLegalIdStart(ch) || (ch >= '0' && ch <= '9');
}

static constexpr char NormalizeIdChar(char32_t ch) {
  return static_cast<char>(ch >= 'A' && ch <= 'Z' ? ch - 'A' + 'a' : ch);
}

static bool GetLowerCaseName(
    IoStatementState &io, char buffer[], std::size_t maxLength) {
  std::size_t byteLength{0};
  if (auto ch{io.GetNextNonBlank(byteLength)}) {
    if (IsLegalIdStart(*ch)) {
      std::size_t j{0};
      do {
        buffer[j] = NormalizeIdChar(*ch);
        io.HandleRelativePosition(byteLength);
        ch = io.GetCurrentChar(byteLength);
      } while (++j < maxLength && ch && IsLegalIdChar(*ch));
      buffer[j++] = '\0';
      if (j <= maxLength) {
        return true;
      }
      io.GetIoErrorHandler().SignalError(
          "Identifier '%s...' in NAMELIST input group is too long", buffer);
    }
  }
  return false;
}

static std::optional<SubscriptValue> GetSubscriptValue(IoStatementState &io) {
  std::optional<SubscriptValue> value;
  std::size_t byteCount{0};
  std::optional<char32_t> ch{io.GetCurrentChar(byteCount)};
  bool negate{ch && *ch == '-'};
  if ((ch && *ch == '+') || negate) {
    io.HandleRelativePosition(byteCount);
    ch = io.GetCurrentChar(byteCount);
  }
  bool overflow{false};
  while (ch && *ch >= '0' && *ch <= '9') {
    SubscriptValue was{value.value_or(0)};
    overflow |= was >= std::numeric_limits<SubscriptValue>::max() / 10;
    value = 10 * was + *ch - '0';
    io.HandleRelativePosition(byteCount);
    ch = io.GetCurrentChar(byteCount);
  }
  if (overflow) {
    io.GetIoErrorHandler().SignalError(
        "NAMELIST input subscript value overflow");
    return std::nullopt;
  }
  if (negate) {
    if (value) {
      return -*value;
    } else {
      io.HandleRelativePosition(-byteCount); // give back '-' with no digits
    }
  }
  return value;
}

static bool HandleSubscripts(IoStatementState &io, Descriptor &desc,
    const Descriptor &source, const char *name) {
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  // Allow for blanks in subscripts; they're nonstandard, but not
  // ambiguous within the parentheses.
  SubscriptValue lower[maxRank], upper[maxRank], stride[maxRank];
  int j{0};
  std::size_t contiguousStride{source.ElementBytes()};
  bool ok{true};
  std::size_t byteCount{0};
  std::optional<char32_t> ch{io.GetNextNonBlank(byteCount)};
  char32_t comma{GetComma(io)};
  for (; ch && *ch != ')'; ++j) {
    SubscriptValue dimLower{0}, dimUpper{0}, dimStride{0};
    if (j < maxRank && j < source.rank()) {
      const Dimension &dim{source.GetDimension(j)};
      dimLower = dim.LowerBound();
      dimUpper = dim.UpperBound();
      dimStride =
          dim.ByteStride() / std::max<SubscriptValue>(contiguousStride, 1);
      contiguousStride *= dim.Extent();
    } else if (ok) {
      handler.SignalError(
          "Too many subscripts for rank-%d NAMELIST group item '%s'",
          source.rank(), name);
      ok = false;
    }
    if (auto low{GetSubscriptValue(io)}) {
      if (*low < dimLower || (dimUpper >= dimLower && *low > dimUpper)) {
        if (ok) {
          handler.SignalError("Subscript %jd out of range %jd..%jd in NAMELIST "
                              "group item '%s' dimension %d",
              static_cast<std::intmax_t>(*low),
              static_cast<std::intmax_t>(dimLower),
              static_cast<std::intmax_t>(dimUpper), name, j + 1);
          ok = false;
        }
      } else {
        dimLower = *low;
      }
      ch = io.GetNextNonBlank(byteCount);
    }
    if (ch && *ch == ':') {
      io.HandleRelativePosition(byteCount);
      ch = io.GetNextNonBlank(byteCount);
      if (auto high{GetSubscriptValue(io)}) {
        if (*high > dimUpper) {
          if (ok) {
            handler.SignalError(
                "Subscript triplet upper bound %jd out of range (>%jd) in "
                "NAMELIST group item '%s' dimension %d",
                static_cast<std::intmax_t>(*high),
                static_cast<std::intmax_t>(dimUpper), name, j + 1);
            ok = false;
          }
        } else {
          dimUpper = *high;
        }
        ch = io.GetNextNonBlank(byteCount);
      }
      if (ch && *ch == ':') {
        io.HandleRelativePosition(byteCount);
        ch = io.GetNextNonBlank(byteCount);
        if (auto str{GetSubscriptValue(io)}) {
          dimStride = *str;
          ch = io.GetNextNonBlank(byteCount);
        }
      }
    } else { // scalar
      dimUpper = dimLower;
      dimStride = 0;
    }
    if (ch && *ch == comma) {
      io.HandleRelativePosition(byteCount);
      ch = io.GetNextNonBlank(byteCount);
    }
    if (ok) {
      lower[j] = dimLower;
      upper[j] = dimUpper;
      stride[j] = dimStride;
    }
  }
  if (ok) {
    if (ch && *ch == ')') {
      io.HandleRelativePosition(byteCount);
      if (desc.EstablishPointerSection(source, lower, upper, stride)) {
        return true;
      } else {
        handler.SignalError(
            "Bad subscripts for NAMELIST input group item '%s'", name);
      }
    } else {
      handler.SignalError(
          "Bad subscripts (missing ')') for NAMELIST input group item '%s'",
          name);
    }
  }
  return false;
}

static bool HandleSubstring(
    IoStatementState &io, Descriptor &desc, const char *name) {
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  auto pair{desc.type().GetCategoryAndKind()};
  if (!pair || pair->first != TypeCategory::Character) {
    handler.SignalError("Substring reference to non-character item '%s'", name);
    return false;
  }
  int kind{pair->second};
  SubscriptValue chars{static_cast<SubscriptValue>(desc.ElementBytes()) / kind};
  // Allow for blanks in substring bounds; they're nonstandard, but not
  // ambiguous within the parentheses.
  std::optional<SubscriptValue> lower, upper;
  std::size_t byteCount{0};
  std::optional<char32_t> ch{io.GetNextNonBlank(byteCount)};
  if (ch) {
    if (*ch == ':') {
      lower = 1;
    } else {
      lower = GetSubscriptValue(io);
      ch = io.GetNextNonBlank(byteCount);
    }
  }
  if (ch && ch == ':') {
    io.HandleRelativePosition(byteCount);
    ch = io.GetNextNonBlank(byteCount);
    if (ch) {
      if (*ch == ')') {
        upper = chars;
      } else {
        upper = GetSubscriptValue(io);
        ch = io.GetNextNonBlank(byteCount);
      }
    }
  }
  if (ch && *ch == ')') {
    io.HandleRelativePosition(byteCount);
    if (lower && upper) {
      if (*lower > *upper) {
        // An empty substring, whatever the values are
        desc.raw().elem_len = 0;
        return true;
      }
      if (*lower >= 1 || *upper <= chars) {
        // Offset the base address & adjust the element byte length
        desc.raw().elem_len = (*upper - *lower + 1) * kind;
        desc.set_base_addr(reinterpret_cast<void *>(
            reinterpret_cast<char *>(desc.raw().base_addr) +
            kind * (*lower - 1)));
        return true;
      }
    }
    handler.SignalError(
        "Bad substring bounds for NAMELIST input group item '%s'", name);
  } else {
    handler.SignalError(
        "Bad substring (missing ')') for NAMELIST input group item '%s'", name);
  }
  return false;
}

static bool HandleComponent(IoStatementState &io, Descriptor &desc,
    const Descriptor &source, const char *name) {
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  char compName[nameBufferSize];
  if (GetLowerCaseName(io, compName, sizeof compName)) {
    const DescriptorAddendum *addendum{source.Addendum()};
    if (const typeInfo::DerivedType *
        type{addendum ? addendum->derivedType() : nullptr}) {
      if (const typeInfo::Component *
          comp{type->FindDataComponent(compName, std::strlen(compName))}) {
        bool createdDesc{false};
        if (comp->rank() > 0 && source.rank() > 0) {
          // If base and component are both arrays, the component name
          // must be followed by subscripts; process them now.
          std::size_t byteCount{0};
          if (std::optional<char32_t> next{io.GetNextNonBlank(byteCount)};
              next && *next == '(') {
            io.HandleRelativePosition(byteCount); // skip over '('
            StaticDescriptor<maxRank, true, 16> staticDesc;
            Descriptor &tmpDesc{staticDesc.descriptor()};
            comp->CreatePointerDescriptor(tmpDesc, source, handler);
            if (!HandleSubscripts(io, desc, tmpDesc, compName)) {
              return false;
            }
            createdDesc = true;
          }
        }
        if (!createdDesc) {
          comp->CreatePointerDescriptor(desc, source, handler);
        }
        if (source.rank() > 0) {
          if (desc.rank() > 0) {
            handler.SignalError(
                "NAMELIST component reference '%%%s' of input group "
                "item %s cannot be an array when its base is not scalar",
                compName, name);
            return false;
          }
          desc.raw().rank = source.rank();
          for (int j{0}; j < source.rank(); ++j) {
            const auto &srcDim{source.GetDimension(j)};
            desc.GetDimension(j)
                .SetBounds(1, srcDim.UpperBound())
                .SetByteStride(srcDim.ByteStride());
          }
        }
        return true;
      } else {
        handler.SignalError(
            "NAMELIST component reference '%%%s' of input group item %s is not "
            "a component of its derived type",
            compName, name);
      }
    } else if (source.type().IsDerived()) {
      handler.Crash("Derived type object '%s' in NAMELIST is missing its "
                    "derived type information!",
          name);
    } else {
      handler.SignalError("NAMELIST component reference '%%%s' of input group "
                          "item %s for non-derived type",
          compName, name);
    }
  } else {
    handler.SignalError("NAMELIST component reference of input group item %s "
                        "has no name after '%%'",
        name);
  }
  return false;
}

// Advance to the terminal '/' of a namelist group.
static void SkipNamelistGroup(IoStatementState &io) {
  std::size_t byteCount{0};
  while (auto ch{io.GetNextNonBlank(byteCount)}) {
    io.HandleRelativePosition(byteCount);
    if (*ch == '/') {
      break;
    } else if (*ch == '\'' || *ch == '"') {
      // Skip quoted character literal
      char32_t quote{*ch};
      while (true) {
        if ((ch = io.GetCurrentChar(byteCount))) {
          io.HandleRelativePosition(byteCount);
          if (*ch == quote) {
            break;
          }
        } else if (!io.AdvanceRecord()) {
          return;
        }
      }
    }
  }
}

bool IONAME(InputNamelist)(Cookie cookie, const NamelistGroup &group) {
  IoStatementState &io{*cookie};
  io.CheckFormattedStmtType<Direction::Input>("InputNamelist");
  io.mutableModes().inNamelist = true;
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  auto *listInput{io.get_if<ListDirectedStatementState<Direction::Input>>()};
  RUNTIME_CHECK(handler, listInput != nullptr);
  // Find this namelist group's header in the input
  io.BeginReadingRecord();
  std::optional<char32_t> next;
  char name[nameBufferSize];
  RUNTIME_CHECK(handler, group.groupName != nullptr);
  char32_t comma{GetComma(io)};
  std::size_t byteCount{0};
  while (true) {
    next = io.GetNextNonBlank(byteCount);
    while (next && *next != '&') {
      // Extension: comment lines without ! before namelist groups
      if (!io.AdvanceRecord()) {
        next.reset();
      } else {
        next = io.GetNextNonBlank(byteCount);
      }
    }
    if (!next) {
      handler.SignalEnd();
      return false;
    }
    if (*next != '&') {
      handler.SignalError(
          "NAMELIST input group does not begin with '&' (at '%lc')", *next);
      return false;
    }
    io.HandleRelativePosition(byteCount);
    if (!GetLowerCaseName(io, name, sizeof name)) {
      handler.SignalError("NAMELIST input group has no name");
      return false;
    }
    if (std::strcmp(group.groupName, name) == 0) {
      break; // found it
    }
    SkipNamelistGroup(io);
  }
  // Read the group's items
  while (true) {
    next = io.GetNextNonBlank(byteCount);
    if (!next || *next == '/') {
      break;
    }
    if (!GetLowerCaseName(io, name, sizeof name)) {
      handler.SignalError(
          "NAMELIST input group '%s' was not terminated at '%c'",
          group.groupName, static_cast<char>(*next));
      return false;
    }
    std::size_t itemIndex{0};
    for (; itemIndex < group.items; ++itemIndex) {
      if (std::strcmp(name, group.item[itemIndex].name) == 0) {
        break;
      }
    }
    if (itemIndex >= group.items) {
      handler.SignalError(
          "'%s' is not an item in NAMELIST group '%s'", name, group.groupName);
      return false;
    }
    // Handle indexing and components, if any.  No spaces are allowed.
    // A copy of the descriptor is made if necessary.
    const Descriptor &itemDescriptor{group.item[itemIndex].descriptor};
    const Descriptor *useDescriptor{&itemDescriptor};
    StaticDescriptor<maxRank, true, 16> staticDesc[2];
    int whichStaticDesc{0};
    next = io.GetCurrentChar(byteCount);
    bool hadSubscripts{false};
    bool hadSubstring{false};
    if (next && (*next == '(' || *next == '%')) {
      do {
        Descriptor &mutableDescriptor{staticDesc[whichStaticDesc].descriptor()};
        whichStaticDesc ^= 1;
        io.HandleRelativePosition(byteCount); // skip over '(' or '%'
        if (*next == '(') {
          if (!hadSubstring && (hadSubscripts || useDescriptor->rank() == 0)) {
            mutableDescriptor = *useDescriptor;
            mutableDescriptor.raw().attribute = CFI_attribute_pointer;
            if (!HandleSubstring(io, mutableDescriptor, name)) {
              return false;
            }
            hadSubstring = true;
          } else if (hadSubscripts) {
            handler.SignalError("Multiple sets of subscripts for item '%s' in "
                                "NAMELIST group '%s'",
                name, group.groupName);
            return false;
          } else {
            if (!HandleSubscripts(
                    io, mutableDescriptor, *useDescriptor, name)) {
              return false;
            }
          }
          hadSubscripts = true;
        } else {
          if (!HandleComponent(io, mutableDescriptor, *useDescriptor, name)) {
            return false;
          }
          hadSubscripts = false;
          hadSubstring = false;
        }
        useDescriptor = &mutableDescriptor;
        next = io.GetCurrentChar(byteCount);
      } while (next && (*next == '(' || *next == '%'));
    }
    // Skip the '='
    next = io.GetNextNonBlank(byteCount);
    if (!next || *next != '=') {
      handler.SignalError("No '=' found after item '%s' in NAMELIST group '%s'",
          name, group.groupName);
      return false;
    }
    io.HandleRelativePosition(byteCount);
    // Read the values into the descriptor.  An array can be short.
    listInput->ResetForNextNamelistItem(useDescriptor->rank() > 0);
    if (!descr::DescriptorIO<Direction::Input>(io, *useDescriptor)) {
      return false;
    }
    next = io.GetNextNonBlank(byteCount);
    if (next && *next == comma) {
      io.HandleRelativePosition(byteCount);
    }
  }
  if (!next || *next != '/') {
    handler.SignalError(
        "No '/' found after NAMELIST group '%s'", group.groupName);
    return false;
  }
  io.HandleRelativePosition(byteCount);
  return true;
}

bool IsNamelistNameOrSlash(IoStatementState &io) {
  if (auto *listInput{
          io.get_if<ListDirectedStatementState<Direction::Input>>()}) {
    if (listInput->inNamelistArray()) {
      SavedPosition savedPosition{io};
      std::size_t byteCount{0};
      if (auto ch{io.GetNextNonBlank(byteCount)}) {
        if (IsLegalIdStart(*ch)) {
          do {
            io.HandleRelativePosition(byteCount);
            ch = io.GetCurrentChar(byteCount);
          } while (ch && IsLegalIdChar(*ch));
          ch = io.GetNextNonBlank(byteCount);
          // TODO: how to deal with NaN(...) ambiguity?
          return ch && (*ch == '=' || *ch == '(' || *ch == '%');
        } else {
          return *ch == '/';
        }
      }
    }
  }
  return false;
}

} // namespace Fortran::runtime::io
