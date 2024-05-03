//===-- runtime/character.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/character.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Common/bit-population-count.h"
#include "flang/Common/uint128.h"
#include "flang/Runtime/character.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"
#include <algorithm>
#include <cstring>

namespace Fortran::runtime {

template <typename CHAR>
inline RT_API_ATTRS int CompareToBlankPadding(
    const CHAR *x, std::size_t chars) {
  using UNSIGNED_CHAR = std::make_unsigned_t<CHAR>;
  const auto blank{static_cast<UNSIGNED_CHAR>(' ')};
  for (; chars-- > 0; ++x) {
    const UNSIGNED_CHAR ux{*reinterpret_cast<const UNSIGNED_CHAR *>(x)};
    if (ux < blank) {
      return -1;
    }
    if (ux > blank) {
      return 1;
    }
  }
  return 0;
}

RT_OFFLOAD_API_GROUP_BEGIN

template <typename CHAR>
RT_API_ATTRS int CharacterScalarCompare(
    const CHAR *x, const CHAR *y, std::size_t xChars, std::size_t yChars) {
  auto minChars{std::min(xChars, yChars)};
  if constexpr (sizeof(CHAR) == 1) {
    // don't use for kind=2 or =4, that would fail on little-endian machines
    int cmp{Fortran::runtime::memcmp(x, y, minChars)};
    if (cmp < 0) {
      return -1;
    }
    if (cmp > 0) {
      return 1;
    }
    if (xChars == yChars) {
      return 0;
    }
    x += minChars;
    y += minChars;
  } else {
    for (std::size_t n{minChars}; n-- > 0; ++x, ++y) {
      if (*x < *y) {
        return -1;
      }
      if (*x > *y) {
        return 1;
      }
    }
  }
  if (int cmp{CompareToBlankPadding(x, xChars - minChars)}) {
    return cmp;
  }
  return -CompareToBlankPadding(y, yChars - minChars);
}

template RT_API_ATTRS int CharacterScalarCompare<char>(
    const char *x, const char *y, std::size_t xChars, std::size_t yChars);
template RT_API_ATTRS int CharacterScalarCompare<char16_t>(const char16_t *x,
    const char16_t *y, std::size_t xChars, std::size_t yChars);
template RT_API_ATTRS int CharacterScalarCompare<char32_t>(const char32_t *x,
    const char32_t *y, std::size_t xChars, std::size_t yChars);

RT_OFFLOAD_API_GROUP_END

// Shift count to use when converting between character lengths
// and byte counts.
template <typename CHAR>
constexpr int shift{common::TrailingZeroBitCount(sizeof(CHAR))};

template <typename CHAR>
static RT_API_ATTRS void Compare(Descriptor &result, const Descriptor &x,
    const Descriptor &y, const Terminator &terminator) {
  RUNTIME_CHECK(
      terminator, x.rank() == y.rank() || x.rank() == 0 || y.rank() == 0);
  int rank{std::max(x.rank(), y.rank())};
  SubscriptValue ub[maxRank], xAt[maxRank], yAt[maxRank];
  SubscriptValue elements{1};
  for (int j{0}; j < rank; ++j) {
    if (x.rank() > 0 && y.rank() > 0) {
      SubscriptValue xUB{x.GetDimension(j).Extent()};
      SubscriptValue yUB{y.GetDimension(j).Extent()};
      if (xUB != yUB) {
        terminator.Crash("Character array comparison: operands are not "
                         "conforming on dimension %d (%jd != %jd)",
            j + 1, static_cast<std::intmax_t>(xUB),
            static_cast<std::intmax_t>(yUB));
      }
      ub[j] = xUB;
    } else {
      ub[j] = (x.rank() ? x : y).GetDimension(j).Extent();
    }
    elements *= ub[j];
  }
  x.GetLowerBounds(xAt);
  y.GetLowerBounds(yAt);
  result.Establish(
      TypeCategory::Logical, 1, nullptr, rank, ub, CFI_attribute_allocatable);
  for (int j{0}; j < rank; ++j) {
    result.GetDimension(j).SetBounds(1, ub[j]);
  }
  if (result.Allocate() != CFI_SUCCESS) {
    terminator.Crash("Compare: could not allocate storage for result");
  }
  std::size_t xChars{x.ElementBytes() >> shift<CHAR>};
  std::size_t yChars{y.ElementBytes() >> shift<char>};
  for (SubscriptValue resultAt{0}; elements-- > 0;
       ++resultAt, x.IncrementSubscripts(xAt), y.IncrementSubscripts(yAt)) {
    *result.OffsetElement<char>(resultAt) = CharacterScalarCompare<CHAR>(
        x.Element<CHAR>(xAt), y.Element<CHAR>(yAt), xChars, yChars);
  }
}

template <typename CHAR, bool ADJUSTR>
static RT_API_ATTRS void Adjust(CHAR *to, const CHAR *from, std::size_t chars) {
  if constexpr (ADJUSTR) {
    std::size_t j{chars}, k{chars};
    for (; k > 0 && from[k - 1] == ' '; --k) {
    }
    while (k > 0) {
      to[--j] = from[--k];
    }
    while (j > 0) {
      to[--j] = ' ';
    }
  } else { // ADJUSTL
    std::size_t j{0}, k{0};
    for (; k < chars && from[k] == ' '; ++k) {
    }
    while (k < chars) {
      to[j++] = from[k++];
    }
    while (j < chars) {
      to[j++] = ' ';
    }
  }
}

template <typename CHAR, bool ADJUSTR>
static RT_API_ATTRS void AdjustLRHelper(Descriptor &result,
    const Descriptor &string, const Terminator &terminator) {
  int rank{string.rank()};
  SubscriptValue ub[maxRank], stringAt[maxRank];
  SubscriptValue elements{1};
  for (int j{0}; j < rank; ++j) {
    ub[j] = string.GetDimension(j).Extent();
    elements *= ub[j];
    stringAt[j] = 1;
  }
  string.GetLowerBounds(stringAt);
  std::size_t elementBytes{string.ElementBytes()};
  result.Establish(string.type(), elementBytes, nullptr, rank, ub,
      CFI_attribute_allocatable);
  for (int j{0}; j < rank; ++j) {
    result.GetDimension(j).SetBounds(1, ub[j]);
  }
  if (result.Allocate() != CFI_SUCCESS) {
    terminator.Crash("ADJUSTL/R: could not allocate storage for result");
  }
  for (SubscriptValue resultAt{0}; elements-- > 0;
       resultAt += elementBytes, string.IncrementSubscripts(stringAt)) {
    Adjust<CHAR, ADJUSTR>(result.OffsetElement<CHAR>(resultAt),
        string.Element<const CHAR>(stringAt), elementBytes >> shift<CHAR>);
  }
}

template <bool ADJUSTR>
RT_API_ATTRS void AdjustLR(Descriptor &result, const Descriptor &string,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  switch (string.raw().type) {
  case CFI_type_char:
    AdjustLRHelper<char, ADJUSTR>(result, string, terminator);
    break;
  case CFI_type_char16_t:
    AdjustLRHelper<char16_t, ADJUSTR>(result, string, terminator);
    break;
  case CFI_type_char32_t:
    AdjustLRHelper<char32_t, ADJUSTR>(result, string, terminator);
    break;
  default:
    terminator.Crash("ADJUSTL/R: bad string type code %d",
        static_cast<int>(string.raw().type));
  }
}

template <typename CHAR>
inline RT_API_ATTRS std::size_t LenTrim(const CHAR *x, std::size_t chars) {
  while (chars > 0 && x[chars - 1] == ' ') {
    --chars;
  }
  return chars;
}

template <typename INT, typename CHAR>
static RT_API_ATTRS void LenTrim(Descriptor &result, const Descriptor &string,
    const Terminator &terminator) {
  int rank{string.rank()};
  SubscriptValue ub[maxRank], stringAt[maxRank];
  SubscriptValue elements{1};
  for (int j{0}; j < rank; ++j) {
    ub[j] = string.GetDimension(j).Extent();
    elements *= ub[j];
  }
  string.GetLowerBounds(stringAt);
  result.Establish(TypeCategory::Integer, sizeof(INT), nullptr, rank, ub,
      CFI_attribute_allocatable);
  for (int j{0}; j < rank; ++j) {
    result.GetDimension(j).SetBounds(1, ub[j]);
  }
  if (result.Allocate() != CFI_SUCCESS) {
    terminator.Crash("LEN_TRIM: could not allocate storage for result");
  }
  std::size_t stringElementChars{string.ElementBytes() >> shift<CHAR>};
  for (SubscriptValue resultAt{0}; elements-- > 0;
       resultAt += sizeof(INT), string.IncrementSubscripts(stringAt)) {
    *result.OffsetElement<INT>(resultAt) =
        LenTrim(string.Element<CHAR>(stringAt), stringElementChars);
  }
}

template <typename CHAR>
static RT_API_ATTRS void LenTrimKind(Descriptor &result,
    const Descriptor &string, int kind, const Terminator &terminator) {
  switch (kind) {
  case 1:
    LenTrim<CppTypeFor<TypeCategory::Integer, 1>, CHAR>(
        result, string, terminator);
    break;
  case 2:
    LenTrim<CppTypeFor<TypeCategory::Integer, 2>, CHAR>(
        result, string, terminator);
    break;
  case 4:
    LenTrim<CppTypeFor<TypeCategory::Integer, 4>, CHAR>(
        result, string, terminator);
    break;
  case 8:
    LenTrim<CppTypeFor<TypeCategory::Integer, 8>, CHAR>(
        result, string, terminator);
    break;
  case 16:
    LenTrim<CppTypeFor<TypeCategory::Integer, 16>, CHAR>(
        result, string, terminator);
    break;
  default:
    terminator.Crash(
        "not yet implemented: CHARACTER(KIND=%d) in LEN_TRIM intrinsic", kind);
  }
}

// INDEX implementation
template <typename CHAR>
inline RT_API_ATTRS std::size_t Index(const CHAR *x, std::size_t xLen,
    const CHAR *want, std::size_t wantLen, bool back) {
  if (xLen < wantLen) {
    return 0;
  }
  if (xLen == 0) {
    return 1; // wantLen is also 0, so trivial match
  }
  if (back) {
    // If wantLen==0, returns xLen + 1 per standard (and all other compilers)
    std::size_t at{xLen - wantLen + 1};
    for (; at > 0; --at) {
      std::size_t j{1};
      for (; j <= wantLen; ++j) {
        if (x[at + j - 2] != want[j - 1]) {
          break;
        }
      }
      if (j > wantLen) {
        return at;
      }
    }
    return 0;
  }
  // Non-trivial forward substring search: use a simplified form of
  // Boyer-Moore substring searching.
  for (std::size_t at{1}; at + wantLen - 1 <= xLen;) {
    // Compare x(at:at+wantLen-1) with want(1:wantLen).
    // The comparison proceeds from the ends of the substrings forward
    // so that we can skip ahead by multiple positions on a miss.
    std::size_t j{wantLen};
    CHAR ch;
    for (; j > 0; --j) {
      ch = x[at + j - 2];
      if (ch != want[j - 1]) {
        break;
      }
    }
    if (j == 0) {
      return at; // found a match
    }
    // Suppose we have at==2:
    // "THAT FORTRAN THAT I RAN" <- the string (x) in which we search
    //   "THAT I RAN"            <- the string (want) for which we search
    //          ^------------------ j==7, ch=='T'
    // We can shift ahead 3 positions to at==5 to align the 'T's:
    // "THAT FORTRAN THAT I RAN"
    //      "THAT I RAN"
    std::size_t shift{1};
    for (; shift < j; ++shift) {
      if (want[j - shift - 1] == ch) {
        break;
      }
    }
    at += shift;
  }
  return 0;
}

// SCAN and VERIFY implementation help.  These intrinsic functions
// do pretty much the same thing, so they're templatized with a
// distinguishing flag.

enum class CharFunc { Index, Scan, Verify };

template <typename CHAR, CharFunc FUNC>
inline RT_API_ATTRS std::size_t ScanVerify(const CHAR *x, std::size_t xLen,
    const CHAR *set, std::size_t setLen, bool back) {
  std::size_t at{back ? xLen : 1};
  int increment{back ? -1 : 1};
  for (; xLen-- > 0; at += increment) {
    CHAR ch{x[at - 1]};
    bool inSet{false};
    // TODO: If set is sorted, could use binary search
    for (std::size_t j{0}; j < setLen; ++j) {
      if (set[j] == ch) {
        inSet = true;
        break;
      }
    }
    if (inSet != (FUNC == CharFunc::Verify)) {
      return at;
    }
  }
  return 0;
}

// Specialization for one-byte characters
template <bool IS_VERIFY = false>
inline RT_API_ATTRS std::size_t ScanVerify(const char *x, std::size_t xLen,
    const char *set, std::size_t setLen, bool back) {
  std::size_t at{back ? xLen : 1};
  int increment{back ? -1 : 1};
  if (xLen > 0) {
    std::uint64_t bitSet[256 / 64]{0};
    std::uint64_t one{1};
    for (std::size_t j{0}; j < setLen; ++j) {
      unsigned setCh{static_cast<unsigned char>(set[j])};
      bitSet[setCh / 64] |= one << (setCh % 64);
    }
    for (; xLen-- > 0; at += increment) {
      unsigned ch{static_cast<unsigned char>(x[at - 1])};
      bool inSet{((bitSet[ch / 64] >> (ch % 64)) & 1) != 0};
      if (inSet != IS_VERIFY) {
        return at;
      }
    }
  }
  return 0;
}

template <typename INT, typename CHAR, CharFunc FUNC>
static RT_API_ATTRS void GeneralCharFunc(Descriptor &result,
    const Descriptor &string, const Descriptor &arg, const Descriptor *back,
    const Terminator &terminator) {
  int rank{string.rank() ? string.rank()
          : arg.rank()   ? arg.rank()
          : back         ? back->rank()
                         : 0};
  SubscriptValue ub[maxRank], stringAt[maxRank], argAt[maxRank],
      backAt[maxRank];
  SubscriptValue elements{1};
  for (int j{0}; j < rank; ++j) {
    ub[j] = string.rank() ? string.GetDimension(j).Extent()
        : arg.rank()      ? arg.GetDimension(j).Extent()
        : back            ? back->GetDimension(j).Extent()
                          : 1;
    elements *= ub[j];
  }
  string.GetLowerBounds(stringAt);
  arg.GetLowerBounds(argAt);
  if (back) {
    back->GetLowerBounds(backAt);
  }
  result.Establish(TypeCategory::Integer, sizeof(INT), nullptr, rank, ub,
      CFI_attribute_allocatable);
  for (int j{0}; j < rank; ++j) {
    result.GetDimension(j).SetBounds(1, ub[j]);
  }
  if (result.Allocate() != CFI_SUCCESS) {
    terminator.Crash("SCAN/VERIFY: could not allocate storage for result");
  }
  std::size_t stringElementChars{string.ElementBytes() >> shift<CHAR>};
  std::size_t argElementChars{arg.ElementBytes() >> shift<CHAR>};
  for (SubscriptValue resultAt{0}; elements-- > 0; resultAt += sizeof(INT),
       string.IncrementSubscripts(stringAt), arg.IncrementSubscripts(argAt),
       back && back->IncrementSubscripts(backAt)) {
    if constexpr (FUNC == CharFunc::Index) {
      *result.OffsetElement<INT>(resultAt) =
          Index<CHAR>(string.Element<CHAR>(stringAt), stringElementChars,
              arg.Element<CHAR>(argAt), argElementChars,
              back && IsLogicalElementTrue(*back, backAt));
    } else if constexpr (FUNC == CharFunc::Scan) {
      *result.OffsetElement<INT>(resultAt) =
          ScanVerify<CHAR, CharFunc::Scan>(string.Element<CHAR>(stringAt),
              stringElementChars, arg.Element<CHAR>(argAt), argElementChars,
              back && IsLogicalElementTrue(*back, backAt));
    } else if constexpr (FUNC == CharFunc::Verify) {
      *result.OffsetElement<INT>(resultAt) =
          ScanVerify<CHAR, CharFunc::Verify>(string.Element<CHAR>(stringAt),
              stringElementChars, arg.Element<CHAR>(argAt), argElementChars,
              back && IsLogicalElementTrue(*back, backAt));
    } else {
      static_assert(FUNC == CharFunc::Index || FUNC == CharFunc::Scan ||
          FUNC == CharFunc::Verify);
    }
  }
}

template <typename CHAR, CharFunc FUNC>
static RT_API_ATTRS void GeneralCharFuncKind(Descriptor &result,
    const Descriptor &string, const Descriptor &arg, const Descriptor *back,
    int kind, const Terminator &terminator) {
  switch (kind) {
  case 1:
    GeneralCharFunc<CppTypeFor<TypeCategory::Integer, 1>, CHAR, FUNC>(
        result, string, arg, back, terminator);
    break;
  case 2:
    GeneralCharFunc<CppTypeFor<TypeCategory::Integer, 2>, CHAR, FUNC>(
        result, string, arg, back, terminator);
    break;
  case 4:
    GeneralCharFunc<CppTypeFor<TypeCategory::Integer, 4>, CHAR, FUNC>(
        result, string, arg, back, terminator);
    break;
  case 8:
    GeneralCharFunc<CppTypeFor<TypeCategory::Integer, 8>, CHAR, FUNC>(
        result, string, arg, back, terminator);
    break;
  case 16:
    GeneralCharFunc<CppTypeFor<TypeCategory::Integer, 16>, CHAR, FUNC>(
        result, string, arg, back, terminator);
    break;
  default:
    terminator.Crash("not yet implemented: CHARACTER(KIND=%d) in "
                     "INDEX/SCAN/VERIFY intrinsic",
        kind);
  }
}

template <typename CHAR, bool ISMIN>
static RT_API_ATTRS void MaxMinHelper(Descriptor &accumulator,
    const Descriptor &x, const Terminator &terminator) {
  RUNTIME_CHECK(terminator,
      accumulator.rank() == 0 || x.rank() == 0 ||
          accumulator.rank() == x.rank());
  SubscriptValue ub[maxRank], xAt[maxRank];
  SubscriptValue elements{1};
  std::size_t accumChars{accumulator.ElementBytes() >> shift<CHAR>};
  std::size_t xChars{x.ElementBytes() >> shift<CHAR>};
  std::size_t chars{std::max(accumChars, xChars)};
  bool reallocate{accumulator.raw().base_addr == nullptr ||
      accumChars != chars || (accumulator.rank() == 0 && x.rank() > 0)};
  int rank{std::max(accumulator.rank(), x.rank())};
  for (int j{0}; j < rank; ++j) {
    if (x.rank() > 0) {
      ub[j] = x.GetDimension(j).Extent();
      if (accumulator.rank() > 0) {
        SubscriptValue accumExt{accumulator.GetDimension(j).Extent()};
        if (accumExt != ub[j]) {
          terminator.Crash("Character MAX/MIN: operands are not "
                           "conforming on dimension %d (%jd != %jd)",
              j + 1, static_cast<std::intmax_t>(accumExt),
              static_cast<std::intmax_t>(ub[j]));
        }
      }
    } else {
      ub[j] = accumulator.GetDimension(j).Extent();
    }
    elements *= ub[j];
  }
  x.GetLowerBounds(xAt);
  void *old{nullptr};
  const CHAR *accumData{accumulator.OffsetElement<CHAR>()};
  if (reallocate) {
    old = accumulator.raw().base_addr;
    accumulator.set_base_addr(nullptr);
    accumulator.raw().elem_len = chars << shift<CHAR>;
    for (int j{0}; j < rank; ++j) {
      accumulator.GetDimension(j).SetBounds(1, ub[j]);
    }
    RUNTIME_CHECK(terminator, accumulator.Allocate() == CFI_SUCCESS);
  }
  for (CHAR *result{accumulator.OffsetElement<CHAR>()}; elements-- > 0;
       accumData += accumChars, result += chars, x.IncrementSubscripts(xAt)) {
    const CHAR *xData{x.Element<CHAR>(xAt)};
    int cmp{CharacterScalarCompare(accumData, xData, accumChars, xChars)};
    if constexpr (ISMIN) {
      cmp = -cmp;
    }
    if (cmp < 0) {
      CopyAndPad(result, xData, chars, xChars);
    } else if (result != accumData) {
      CopyAndPad(result, accumData, chars, accumChars);
    }
  }
  FreeMemory(old);
}

template <bool ISMIN>
static RT_API_ATTRS void MaxMin(Descriptor &accumulator, const Descriptor &x,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  RUNTIME_CHECK(terminator, accumulator.raw().type == x.raw().type);
  switch (accumulator.raw().type) {
  case CFI_type_char:
    MaxMinHelper<char, ISMIN>(accumulator, x, terminator);
    break;
  case CFI_type_char16_t:
    MaxMinHelper<char16_t, ISMIN>(accumulator, x, terminator);
    break;
  case CFI_type_char32_t:
    MaxMinHelper<char32_t, ISMIN>(accumulator, x, terminator);
    break;
  default:
    terminator.Crash(
        "Character MAX/MIN: result does not have a character type");
  }
}

extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(CharacterConcatenate)(Descriptor &accumulator,
    const Descriptor &from, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  RUNTIME_CHECK(terminator,
      accumulator.rank() == 0 || from.rank() == 0 ||
          accumulator.rank() == from.rank());
  int rank{std::max(accumulator.rank(), from.rank())};
  SubscriptValue ub[maxRank], fromAt[maxRank];
  SubscriptValue elements{1};
  for (int j{0}; j < rank; ++j) {
    if (accumulator.rank() > 0 && from.rank() > 0) {
      ub[j] = accumulator.GetDimension(j).Extent();
      SubscriptValue fromUB{from.GetDimension(j).Extent()};
      if (ub[j] != fromUB) {
        terminator.Crash("Character array concatenation: operands are not "
                         "conforming on dimension %d (%jd != %jd)",
            j + 1, static_cast<std::intmax_t>(ub[j]),
            static_cast<std::intmax_t>(fromUB));
      }
    } else {
      ub[j] =
          (accumulator.rank() ? accumulator : from).GetDimension(j).Extent();
    }
    elements *= ub[j];
  }
  std::size_t oldBytes{accumulator.ElementBytes()};
  void *old{accumulator.raw().base_addr};
  accumulator.set_base_addr(nullptr);
  std::size_t fromBytes{from.ElementBytes()};
  accumulator.raw().elem_len += fromBytes;
  std::size_t newBytes{accumulator.ElementBytes()};
  for (int j{0}; j < rank; ++j) {
    accumulator.GetDimension(j).SetBounds(1, ub[j]);
  }
  if (accumulator.Allocate() != CFI_SUCCESS) {
    terminator.Crash(
        "CharacterConcatenate: could not allocate storage for result");
  }
  const char *p{static_cast<const char *>(old)};
  char *to{static_cast<char *>(accumulator.raw().base_addr)};
  from.GetLowerBounds(fromAt);
  for (; elements-- > 0;
       to += newBytes, p += oldBytes, from.IncrementSubscripts(fromAt)) {
    std::memcpy(to, p, oldBytes);
    std::memcpy(to + oldBytes, from.Element<char>(fromAt), fromBytes);
  }
  FreeMemory(old);
}

void RTDEF(CharacterConcatenateScalar1)(
    Descriptor &accumulator, const char *from, std::size_t chars) {
  Terminator terminator{__FILE__, __LINE__};
  RUNTIME_CHECK(terminator, accumulator.rank() == 0);
  void *old{accumulator.raw().base_addr};
  accumulator.set_base_addr(nullptr);
  std::size_t oldLen{accumulator.ElementBytes()};
  accumulator.raw().elem_len += chars;
  RUNTIME_CHECK(terminator, accumulator.Allocate() == CFI_SUCCESS);
  std::memcpy(accumulator.OffsetElement<char>(oldLen), from, chars);
  FreeMemory(old);
}

int RTDEF(CharacterCompareScalar)(const Descriptor &x, const Descriptor &y) {
  Terminator terminator{__FILE__, __LINE__};
  RUNTIME_CHECK(terminator, x.rank() == 0);
  RUNTIME_CHECK(terminator, y.rank() == 0);
  RUNTIME_CHECK(terminator, x.raw().type == y.raw().type);
  switch (x.raw().type) {
  case CFI_type_char:
    return CharacterScalarCompare<char>(x.OffsetElement<char>(),
        y.OffsetElement<char>(), x.ElementBytes(), y.ElementBytes());
  case CFI_type_char16_t:
    return CharacterScalarCompare<char16_t>(x.OffsetElement<char16_t>(),
        y.OffsetElement<char16_t>(), x.ElementBytes() >> 1,
        y.ElementBytes() >> 1);
  case CFI_type_char32_t:
    return CharacterScalarCompare<char32_t>(x.OffsetElement<char32_t>(),
        y.OffsetElement<char32_t>(), x.ElementBytes() >> 2,
        y.ElementBytes() >> 2);
  default:
    terminator.Crash("CharacterCompareScalar: bad string type code %d",
        static_cast<int>(x.raw().type));
  }
  return 0;
}

int RTDEF(CharacterCompareScalar1)(
    const char *x, const char *y, std::size_t xChars, std::size_t yChars) {
  return CharacterScalarCompare(x, y, xChars, yChars);
}

int RTDEF(CharacterCompareScalar2)(const char16_t *x, const char16_t *y,
    std::size_t xChars, std::size_t yChars) {
  return CharacterScalarCompare(x, y, xChars, yChars);
}

int RTDEF(CharacterCompareScalar4)(const char32_t *x, const char32_t *y,
    std::size_t xChars, std::size_t yChars) {
  return CharacterScalarCompare(x, y, xChars, yChars);
}

void RTDEF(CharacterCompare)(
    Descriptor &result, const Descriptor &x, const Descriptor &y) {
  Terminator terminator{__FILE__, __LINE__};
  RUNTIME_CHECK(terminator, x.raw().type == y.raw().type);
  switch (x.raw().type) {
  case CFI_type_char:
    Compare<char>(result, x, y, terminator);
    break;
  case CFI_type_char16_t:
    Compare<char16_t>(result, x, y, terminator);
    break;
  case CFI_type_char32_t:
    Compare<char32_t>(result, x, y, terminator);
    break;
  default:
    terminator.Crash("CharacterCompareScalar: bad string type code %d",
        static_cast<int>(x.raw().type));
  }
}

std::size_t RTDEF(CharacterAppend1)(char *lhs, std::size_t lhsBytes,
    std::size_t offset, const char *rhs, std::size_t rhsBytes) {
  if (auto n{std::min(lhsBytes - offset, rhsBytes)}) {
    std::memcpy(lhs + offset, rhs, n);
    offset += n;
  }
  return offset;
}

void RTDEF(CharacterPad1)(char *lhs, std::size_t bytes, std::size_t offset) {
  if (bytes > offset) {
    std::memset(lhs + offset, ' ', bytes - offset);
  }
}

// Intrinsic function entry points

void RTDEF(Adjustl)(Descriptor &result, const Descriptor &string,
    const char *sourceFile, int sourceLine) {
  AdjustLR<false>(result, string, sourceFile, sourceLine);
}

void RTDEF(Adjustr)(Descriptor &result, const Descriptor &string,
    const char *sourceFile, int sourceLine) {
  AdjustLR<true>(result, string, sourceFile, sourceLine);
}

std::size_t RTDEF(Index1)(const char *x, std::size_t xLen, const char *set,
    std::size_t setLen, bool back) {
  return Index<char>(x, xLen, set, setLen, back);
}
std::size_t RTDEF(Index2)(const char16_t *x, std::size_t xLen,
    const char16_t *set, std::size_t setLen, bool back) {
  return Index<char16_t>(x, xLen, set, setLen, back);
}
std::size_t RTDEF(Index4)(const char32_t *x, std::size_t xLen,
    const char32_t *set, std::size_t setLen, bool back) {
  return Index<char32_t>(x, xLen, set, setLen, back);
}

void RTDEF(Index)(Descriptor &result, const Descriptor &string,
    const Descriptor &substring, const Descriptor *back, int kind,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  switch (string.raw().type) {
  case CFI_type_char:
    GeneralCharFuncKind<char, CharFunc::Index>(
        result, string, substring, back, kind, terminator);
    break;
  case CFI_type_char16_t:
    GeneralCharFuncKind<char16_t, CharFunc::Index>(
        result, string, substring, back, kind, terminator);
    break;
  case CFI_type_char32_t:
    GeneralCharFuncKind<char32_t, CharFunc::Index>(
        result, string, substring, back, kind, terminator);
    break;
  default:
    terminator.Crash(
        "INDEX: bad string type code %d", static_cast<int>(string.raw().type));
  }
}

std::size_t RTDEF(LenTrim1)(const char *x, std::size_t chars) {
  return LenTrim(x, chars);
}
std::size_t RTDEF(LenTrim2)(const char16_t *x, std::size_t chars) {
  return LenTrim(x, chars);
}
std::size_t RTDEF(LenTrim4)(const char32_t *x, std::size_t chars) {
  return LenTrim(x, chars);
}

void RTDEF(LenTrim)(Descriptor &result, const Descriptor &string, int kind,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  switch (string.raw().type) {
  case CFI_type_char:
    LenTrimKind<char>(result, string, kind, terminator);
    break;
  case CFI_type_char16_t:
    LenTrimKind<char16_t>(result, string, kind, terminator);
    break;
  case CFI_type_char32_t:
    LenTrimKind<char32_t>(result, string, kind, terminator);
    break;
  default:
    terminator.Crash("LEN_TRIM: bad string type code %d",
        static_cast<int>(string.raw().type));
  }
}

std::size_t RTDEF(Scan1)(const char *x, std::size_t xLen, const char *set,
    std::size_t setLen, bool back) {
  return ScanVerify<char, CharFunc::Scan>(x, xLen, set, setLen, back);
}
std::size_t RTDEF(Scan2)(const char16_t *x, std::size_t xLen,
    const char16_t *set, std::size_t setLen, bool back) {
  return ScanVerify<char16_t, CharFunc::Scan>(x, xLen, set, setLen, back);
}
std::size_t RTDEF(Scan4)(const char32_t *x, std::size_t xLen,
    const char32_t *set, std::size_t setLen, bool back) {
  return ScanVerify<char32_t, CharFunc::Scan>(x, xLen, set, setLen, back);
}

void RTDEF(Scan)(Descriptor &result, const Descriptor &string,
    const Descriptor &set, const Descriptor *back, int kind,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  switch (string.raw().type) {
  case CFI_type_char:
    GeneralCharFuncKind<char, CharFunc::Scan>(
        result, string, set, back, kind, terminator);
    break;
  case CFI_type_char16_t:
    GeneralCharFuncKind<char16_t, CharFunc::Scan>(
        result, string, set, back, kind, terminator);
    break;
  case CFI_type_char32_t:
    GeneralCharFuncKind<char32_t, CharFunc::Scan>(
        result, string, set, back, kind, terminator);
    break;
  default:
    terminator.Crash(
        "SCAN: bad string type code %d", static_cast<int>(string.raw().type));
  }
}

void RTDEF(Repeat)(Descriptor &result, const Descriptor &string,
    std::int64_t ncopies, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (ncopies < 0) {
    terminator.Crash(
        "REPEAT has negative NCOPIES=%jd", static_cast<std::intmax_t>(ncopies));
  }
  std::size_t origBytes{string.ElementBytes()};
  result.Establish(string.type(), origBytes * ncopies, nullptr, 0, nullptr,
      CFI_attribute_allocatable);
  if (result.Allocate() != CFI_SUCCESS) {
    terminator.Crash("REPEAT could not allocate storage for result");
  }
  const char *from{string.OffsetElement()};
  for (char *to{result.OffsetElement()}; ncopies-- > 0; to += origBytes) {
    std::memcpy(to, from, origBytes);
  }
}

void RTDEF(Trim)(Descriptor &result, const Descriptor &string,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  std::size_t resultBytes{0};
  switch (string.raw().type) {
  case CFI_type_char:
    resultBytes =
        LenTrim(string.OffsetElement<const char>(), string.ElementBytes());
    break;
  case CFI_type_char16_t:
    resultBytes = LenTrim(string.OffsetElement<const char16_t>(),
                      string.ElementBytes() >> 1)
        << 1;
    break;
  case CFI_type_char32_t:
    resultBytes = LenTrim(string.OffsetElement<const char32_t>(),
                      string.ElementBytes() >> 2)
        << 2;
    break;
  default:
    terminator.Crash(
        "TRIM: bad string type code %d", static_cast<int>(string.raw().type));
  }
  result.Establish(string.type(), resultBytes, nullptr, 0, nullptr,
      CFI_attribute_allocatable);
  RUNTIME_CHECK(terminator, result.Allocate() == CFI_SUCCESS);
  std::memcpy(result.OffsetElement(), string.OffsetElement(), resultBytes);
}

std::size_t RTDEF(Verify1)(const char *x, std::size_t xLen, const char *set,
    std::size_t setLen, bool back) {
  return ScanVerify<char, CharFunc::Verify>(x, xLen, set, setLen, back);
}
std::size_t RTDEF(Verify2)(const char16_t *x, std::size_t xLen,
    const char16_t *set, std::size_t setLen, bool back) {
  return ScanVerify<char16_t, CharFunc::Verify>(x, xLen, set, setLen, back);
}
std::size_t RTDEF(Verify4)(const char32_t *x, std::size_t xLen,
    const char32_t *set, std::size_t setLen, bool back) {
  return ScanVerify<char32_t, CharFunc::Verify>(x, xLen, set, setLen, back);
}

void RTDEF(Verify)(Descriptor &result, const Descriptor &string,
    const Descriptor &set, const Descriptor *back, int kind,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  switch (string.raw().type) {
  case CFI_type_char:
    GeneralCharFuncKind<char, CharFunc::Verify>(
        result, string, set, back, kind, terminator);
    break;
  case CFI_type_char16_t:
    GeneralCharFuncKind<char16_t, CharFunc::Verify>(
        result, string, set, back, kind, terminator);
    break;
  case CFI_type_char32_t:
    GeneralCharFuncKind<char32_t, CharFunc::Verify>(
        result, string, set, back, kind, terminator);
    break;
  default:
    terminator.Crash(
        "VERIFY: bad string type code %d", static_cast<int>(string.raw().type));
  }
}

void RTDEF(CharacterMax)(Descriptor &accumulator, const Descriptor &x,
    const char *sourceFile, int sourceLine) {
  MaxMin<false>(accumulator, x, sourceFile, sourceLine);
}

void RTDEF(CharacterMin)(Descriptor &accumulator, const Descriptor &x,
    const char *sourceFile, int sourceLine) {
  MaxMin<true>(accumulator, x, sourceFile, sourceLine);
}

RT_EXT_API_GROUP_END
}
} // namespace Fortran::runtime
