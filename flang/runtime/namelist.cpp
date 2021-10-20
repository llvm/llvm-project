//===-- runtime/namelist.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "namelist.h"
#include "descriptor-io.h"
#include "io-stmt.h"
#include "flang/Runtime/io-api.h"
#include <cstring>
#include <limits>

namespace Fortran::runtime::io {

// Max size of a group, symbol or component identifier that can appear in
// NAMELIST input, plus a byte for NUL termination.
static constexpr std::size_t nameBufferSize{201};

bool IONAME(OutputNamelist)(Cookie cookie, const NamelistGroup &group) {
  IoStatementState &io{*cookie};
  io.CheckFormattedStmtType<Direction::Output>("OutputNamelist");
  ConnectionState &connection{io.GetConnectionState()};
  connection.modes.inNamelist = true;
  // Internal functions to advance records and convert case
  const auto EmitWithAdvance{[&](char ch) -> bool {
    return (!connection.NeedAdvance(1) || io.AdvanceRecord()) &&
        io.Emit(&ch, 1);
  }};
  const auto EmitUpperCase{[&](const char *str) -> bool {
    if (connection.NeedAdvance(std::strlen(str)) &&
        !(io.AdvanceRecord() && io.Emit(" ", 1))) {
      return false;
    }
    for (; *str; ++str) {
      char up{*str >= 'a' && *str <= 'z' ? static_cast<char>(*str - 'a' + 'A')
                                         : *str};
      if (!io.Emit(&up, 1)) {
        return false;
      }
    }
    return true;
  }};
  // &GROUP
  if (!(EmitWithAdvance('&') && EmitUpperCase(group.groupName))) {
    return false;
  }
  for (std::size_t j{0}; j < group.items; ++j) {
    // [,]ITEM=...
    const NamelistGroup::Item &item{group.item[j]};
    if (!(EmitWithAdvance(j == 0 ? ' ' : ',') && EmitUpperCase(item.name) &&
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
  if (auto ch{io.GetNextNonBlank()}) {
    if (IsLegalIdStart(*ch)) {
      std::size_t j{0};
      do {
        buffer[j] = NormalizeIdChar(*ch);
        io.HandleRelativePosition(1);
        ch = io.GetCurrentChar();
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
  std::optional<char32_t> ch{io.GetCurrentChar()};
  bool negate{ch && *ch == '-'};
  if (negate) {
    io.HandleRelativePosition(1);
    ch = io.GetCurrentChar();
  }
  bool overflow{false};
  while (ch && *ch >= '0' && *ch <= '9') {
    SubscriptValue was{value.value_or(0)};
    overflow |= was >= std::numeric_limits<SubscriptValue>::max() / 10;
    value = 10 * was + *ch - '0';
    io.HandleRelativePosition(1);
    ch = io.GetCurrentChar();
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
      io.HandleRelativePosition(-1); // give back '-' with no digits
    }
  }
  return value;
}

static bool HandleSubscripts(IoStatementState &io, Descriptor &desc,
    const Descriptor &source, const char *name) {
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  io.HandleRelativePosition(1); // skip '('
  // Allow for blanks in subscripts; they're nonstandard, but not
  // ambiguous within the parentheses.
  SubscriptValue lower[maxRank], upper[maxRank], stride[maxRank];
  int j{0};
  std::size_t elemLen{source.ElementBytes()};
  bool ok{true};
  std::optional<char32_t> ch{io.GetNextNonBlank()};
  for (; ch && *ch != ')'; ++j) {
    SubscriptValue dimLower{0}, dimUpper{0}, dimStride{0};
    if (j < maxRank && j < source.rank()) {
      const Dimension &dim{source.GetDimension(j)};
      dimLower = dim.LowerBound();
      dimUpper = dim.UpperBound();
      dimStride = elemLen ? dim.ByteStride() / elemLen : 1;
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
      ch = io.GetNextNonBlank();
    }
    if (ch && *ch == ':') {
      io.HandleRelativePosition(1);
      ch = io.GetNextNonBlank();
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
        ch = io.GetNextNonBlank();
      }
      if (ch && *ch == ':') {
        io.HandleRelativePosition(1);
        ch = io.GetNextNonBlank();
        if (auto str{GetSubscriptValue(io)}) {
          dimStride = *str;
          ch = io.GetNextNonBlank();
        }
      }
    } else { // scalar
      dimUpper = dimLower;
      dimStride = 0;
    }
    if (ch && *ch == ',') {
      io.HandleRelativePosition(1);
      ch = io.GetNextNonBlank();
    }
    if (ok) {
      lower[j] = dimLower;
      upper[j] = dimUpper;
      stride[j] = dimStride;
    }
  }
  if (ok) {
    if (ch && *ch == ')') {
      io.HandleRelativePosition(1);
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

static bool HandleComponent(IoStatementState &io, Descriptor &desc,
    const Descriptor &source, const char *name) {
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  io.HandleRelativePosition(1); // skip '%'
  char compName[nameBufferSize];
  if (GetLowerCaseName(io, compName, sizeof compName)) {
    const DescriptorAddendum *addendum{source.Addendum()};
    if (const typeInfo::DerivedType *
        type{addendum ? addendum->derivedType() : nullptr}) {
      if (const typeInfo::Component *
          comp{type->FindDataComponent(compName, std::strlen(compName))}) {
        comp->CreatePointerDescriptor(desc, source, nullptr, handler);
        return true;
      } else {
        handler.SignalError(
            "NAMELIST component reference '%%%s' of input group item %s is not "
            "a component of its derived type",
            compName, name);
      }
    } else {
      handler.SignalError("NAMELIST component reference '%%%s' of input group "
                          "item %s for non-derived type",
          compName, name);
    }
  } else {
    handler.SignalError("NAMELIST component reference of input group item %s "
                        "has no name after '%'",
        name);
  }
  return false;
}

bool IONAME(InputNamelist)(Cookie cookie, const NamelistGroup &group) {
  IoStatementState &io{*cookie};
  io.CheckFormattedStmtType<Direction::Input>("InputNamelist");
  ConnectionState &connection{io.GetConnectionState()};
  connection.modes.inNamelist = true;
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  auto *listInput{io.get_if<ListDirectedStatementState<Direction::Input>>()};
  RUNTIME_CHECK(handler, listInput != nullptr);
  // Check the group header
  io.BeginReadingRecord();
  std::optional<char32_t> next{io.GetNextNonBlank()};
  if (!next || *next != '&') {
    handler.SignalError(
        "NAMELIST input group does not begin with '&' (at '%lc')", *next);
    return false;
  }
  io.HandleRelativePosition(1);
  char name[nameBufferSize];
  if (!GetLowerCaseName(io, name, sizeof name)) {
    handler.SignalError("NAMELIST input group has no name");
    return false;
  }
  RUNTIME_CHECK(handler, group.groupName != nullptr);
  if (std::strcmp(group.groupName, name) != 0) {
    handler.SignalError(
        "NAMELIST input group name '%s' is not the expected '%s'", name,
        group.groupName);
    return false;
  }
  // Read the group's items
  while (true) {
    next = io.GetNextNonBlank();
    if (!next || *next == '/') {
      break;
    }
    if (!GetLowerCaseName(io, name, sizeof name)) {
      handler.SignalError(
          "NAMELIST input group '%s' was not terminated", group.groupName);
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
    next = io.GetCurrentChar();
    if (next && (*next == '(' || *next == '%')) {
      do {
        Descriptor &mutableDescriptor{staticDesc[whichStaticDesc].descriptor()};
        whichStaticDesc ^= 1;
        if (*next == '(') {
          HandleSubscripts(io, mutableDescriptor, *useDescriptor, name);
        } else {
          HandleComponent(io, mutableDescriptor, *useDescriptor, name);
        }
        useDescriptor = &mutableDescriptor;
        next = io.GetCurrentChar();
      } while (next && (*next == '(' || *next == '%'));
    }
    // Skip the '='
    next = io.GetNextNonBlank();
    if (!next || *next != '=') {
      handler.SignalError("No '=' found after item '%s' in NAMELIST group '%s'",
          name, group.groupName);
      return false;
    }
    io.HandleRelativePosition(1);
    // Read the values into the descriptor.  An array can be short.
    listInput->ResetForNextNamelistItem();
    if (!descr::DescriptorIO<Direction::Input>(io, *useDescriptor)) {
      return false;
    }
    next = io.GetNextNonBlank();
    if (next && *next == ',') {
      io.HandleRelativePosition(1);
    }
  }
  if (!next || *next != '/') {
    handler.SignalError(
        "No '/' found after NAMELIST group '%s'", group.groupName);
    return false;
  }
  io.HandleRelativePosition(1);
  return true;
}

bool IsNamelistName(IoStatementState &io) {
  if (io.get_if<ListDirectedStatementState<Direction::Input>>()) {
    ConnectionState &connection{io.GetConnectionState()};
    if (connection.modes.inNamelist) {
      SavedPosition savedPosition{connection};
      if (auto ch{io.GetNextNonBlank()}) {
        if (IsLegalIdStart(*ch)) {
          do {
            io.HandleRelativePosition(1);
            ch = io.GetCurrentChar();
          } while (ch && IsLegalIdChar(*ch));
          ch = io.GetNextNonBlank();
          // TODO: how to deal with NaN(...) ambiguity?
          return ch && (ch == '=' || ch == '(' || ch == '%');
        }
      }
    }
  }
  return false;
}

} // namespace Fortran::runtime::io
