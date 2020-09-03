//===-- runtime/character.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "character.h"
#include "descriptor.h"
#include "terminator.h"
#include "flang/Common/bit-population-count.h"
#include "flang/Common/uint128.h"
#include <algorithm>
#include <cstring>

namespace Fortran::runtime {

template <typename CHAR>
inline int CompareToBlankPadding(const CHAR *x, std::size_t chars) {
  for (; chars-- > 0; ++x) {
    if (*x < ' ') {
      return -1;
    }
    if (*x > ' ') {
      return 1;
    }
  }
  return 0;
}

template <typename CHAR>
static int Compare(
    const CHAR *x, const CHAR *y, std::size_t xChars, std::size_t yChars) {
  auto minChars{std::min(xChars, yChars)};
  if constexpr (sizeof(CHAR) == 1) {
    // don't use for kind=2 or =4, that would fail on little-endian machines
    int cmp{std::memcmp(x, y, minChars)};
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

// Shift count to use when converting between character lengths
// and byte counts.
template <typename CHAR>
constexpr int shift{common::TrailingZeroBitCount(sizeof(CHAR))};

template <typename CHAR>
static void Compare(Descriptor &result, const Descriptor &x,
    const Descriptor &y, const Terminator &terminator) {
  RUNTIME_CHECK(
      terminator, x.rank() == y.rank() || x.rank() == 0 || y.rank() == 0);
  int rank{std::max(x.rank(), y.rank())};
  SubscriptValue lb[maxRank], ub[maxRank], xAt[maxRank], yAt[maxRank];
  SubscriptValue elements{1};
  for (int j{0}; j < rank; ++j) {
    lb[j] = 1;
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
    xAt[j] = yAt[j] = 1;
  }
  result.Establish(TypeCategory::Logical, 1, ub, rank);
  if (result.Allocate(lb, ub) != CFI_SUCCESS) {
    terminator.Crash("Compare: could not allocate storage for result");
  }
  std::size_t xChars{x.ElementBytes() >> shift<CHAR>};
  std::size_t yChars{y.ElementBytes() >> shift<char>};
  for (SubscriptValue resultAt{0}; elements-- > 0;
       ++resultAt, x.IncrementSubscripts(xAt), y.IncrementSubscripts(yAt)) {
    *result.OffsetElement<char>(resultAt) =
        Compare(x.Element<CHAR>(xAt), y.Element<CHAR>(yAt), xChars, yChars);
  }
}

template <typename CHAR, bool ADJUSTR>
static void Adjust(CHAR *to, const CHAR *from, std::size_t chars) {
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
static void AdjustLRHelper(Descriptor &result, const Descriptor &string,
    const Terminator &terminator) {
  int rank{string.rank()};
  SubscriptValue lb[maxRank], ub[maxRank], stringAt[maxRank];
  SubscriptValue elements{1};
  for (int j{0}; j < rank; ++j) {
    lb[j] = 1;
    ub[j] = string.GetDimension(j).Extent();
    elements *= ub[j];
    stringAt[j] = 1;
  }
  std::size_t elementBytes{string.ElementBytes()};
  result.Establish(string.type(), elementBytes, ub, rank);
  if (result.Allocate(lb, ub) != CFI_SUCCESS) {
    terminator.Crash("ADJUSTL/R: could not allocate storage for result");
  }
  for (SubscriptValue resultAt{0}; elements-- > 0;
       resultAt += elementBytes, string.IncrementSubscripts(stringAt)) {
    Adjust<CHAR, ADJUSTR>(result.OffsetElement<CHAR>(resultAt),
        string.Element<const CHAR>(stringAt), elementBytes >> shift<CHAR>);
  }
}

template <bool ADJUSTR>
void AdjustLR(Descriptor &result, const Descriptor &string,
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
inline std::size_t LenTrim(const CHAR *x, std::size_t chars) {
  while (chars > 0 && x[chars - 1] == ' ') {
    --chars;
  }
  return chars;
}

template <typename INT, typename CHAR>
static void LenTrim(Descriptor &result, const Descriptor &string,
    const Terminator &terminator) {
  int rank{string.rank()};
  SubscriptValue lb[maxRank], ub[maxRank], stringAt[maxRank];
  SubscriptValue elements{1};
  for (int j{0}; j < rank; ++j) {
    lb[j] = 1;
    ub[j] = string.GetDimension(j).Extent();
    elements *= ub[j];
    stringAt[j] = 1;
  }
  result.Establish(TypeCategory::Integer, sizeof(INT), ub, rank);
  if (result.Allocate(lb, ub) != CFI_SUCCESS) {
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
static void LenTrimKind(Descriptor &result, const Descriptor &string, int kind,
    const Terminator &terminator) {
  switch (kind) {
  case 1:
    LenTrim<std::int8_t, CHAR>(result, string, terminator);
    break;
  case 2:
    LenTrim<std::int16_t, CHAR>(result, string, terminator);
    break;
  case 4:
    LenTrim<std::int32_t, CHAR>(result, string, terminator);
    break;
  case 8:
    LenTrim<std::int64_t, CHAR>(result, string, terminator);
    break;
  case 16:
    LenTrim<common::uint128_t, CHAR>(result, string, terminator);
    break;
  default:
    terminator.Crash("LEN_TRIM: bad KIND=%d", kind);
  }
}

template <typename TO, typename FROM>
static void CopyAndPad(
    TO *to, const FROM *from, std::size_t toChars, std::size_t fromChars) {
  if constexpr (sizeof(TO) != sizeof(FROM)) {
    std::size_t copyChars{std::min(toChars, fromChars)};
    for (std::size_t j{0}; j < copyChars; ++j) {
      to[j] = from[j];
    }
    for (std::size_t j{copyChars}; j < toChars; ++j) {
      to[j] = static_cast<TO>(' ');
    }
  } else if (toChars <= fromChars) {
    std::memcpy(to, from, toChars * shift<TO>);
  } else {
    std::memcpy(to, from, fromChars * shift<TO>);
    for (std::size_t j{fromChars}; j < toChars; ++j) {
      to[j] = static_cast<TO>(' ');
    }
  }
}

template <typename CHAR, bool ISMIN>
static void MaxMinHelper(Descriptor &accumulator, const Descriptor &x,
    const Terminator &terminator) {
  RUNTIME_CHECK(terminator,
      accumulator.rank() == 0 || x.rank() == 0 ||
          accumulator.rank() == x.rank());
  SubscriptValue lb[maxRank], ub[maxRank], xAt[maxRank];
  SubscriptValue elements{1};
  std::size_t accumChars{accumulator.ElementBytes() >> shift<CHAR>};
  std::size_t xChars{x.ElementBytes() >> shift<CHAR>};
  std::size_t chars{std::max(accumChars, xChars)};
  bool reallocate{accumulator.raw().base_addr == nullptr ||
      accumChars != xChars || (accumulator.rank() == 0 && x.rank() > 0)};
  int rank{std::max(accumulator.rank(), x.rank())};
  for (int j{0}; j < rank; ++j) {
    lb[j] = 1;
    if (x.rank() > 0) {
      ub[j] = x.GetDimension(j).Extent();
      xAt[j] = x.GetDimension(j).LowerBound();
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
      xAt[j] = 1;
    }
    elements *= ub[j];
  }
  void *old{nullptr};
  const CHAR *accumData{accumulator.OffsetElement<CHAR>()};
  if (reallocate) {
    old = accumulator.raw().base_addr;
    accumulator.set_base_addr(nullptr);
    accumulator.raw().elem_len = chars << shift<CHAR>;
    RUNTIME_CHECK(terminator, accumulator.Allocate(lb, ub) == CFI_SUCCESS);
  }
  for (CHAR *result{accumulator.OffsetElement<CHAR>()}; elements-- > 0;
       accumData += accumChars, result += chars, x.IncrementSubscripts(xAt)) {
    const CHAR *xData{x.Element<CHAR>(xAt)};
    int cmp{Compare(accumData, xData, accumChars, xChars)};
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
static void MaxMin(Descriptor &accumulator, const Descriptor &x,
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

void RTNAME(CharacterConcatenate)(Descriptor &accumulator,
    const Descriptor &from, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  RUNTIME_CHECK(terminator,
      accumulator.rank() == 0 || from.rank() == 0 ||
          accumulator.rank() == from.rank());
  int rank{std::max(accumulator.rank(), from.rank())};
  SubscriptValue lb[maxRank], ub[maxRank], fromAt[maxRank];
  SubscriptValue elements{1};
  for (int j{0}; j < rank; ++j) {
    lb[j] = 1;
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
    fromAt[j] = 1;
  }
  std::size_t oldBytes{accumulator.ElementBytes()};
  void *old{accumulator.raw().base_addr};
  accumulator.set_base_addr(nullptr);
  std::size_t fromBytes{from.ElementBytes()};
  accumulator.raw().elem_len += fromBytes;
  std::size_t newBytes{accumulator.ElementBytes()};
  if (accumulator.Allocate(lb, ub) != CFI_SUCCESS) {
    terminator.Crash(
        "CharacterConcatenate: could not allocate storage for result");
  }
  const char *p{static_cast<const char *>(old)};
  char *to{static_cast<char *>(accumulator.raw().base_addr)};
  for (; elements-- > 0;
       to += newBytes, p += oldBytes, from.IncrementSubscripts(fromAt)) {
    std::memcpy(to, p, oldBytes);
    std::memcpy(to + oldBytes, from.Element<char>(fromAt), fromBytes);
  }
  FreeMemory(old);
}

void RTNAME(CharacterConcatenateScalar1)(
    Descriptor &accumulator, const char *from, std::size_t chars) {
  Terminator terminator{__FILE__, __LINE__};
  RUNTIME_CHECK(terminator, accumulator.rank() == 0);
  void *old{accumulator.raw().base_addr};
  accumulator.set_base_addr(nullptr);
  std::size_t oldLen{accumulator.ElementBytes()};
  accumulator.raw().elem_len += chars;
  RUNTIME_CHECK(
      terminator, accumulator.Allocate(nullptr, nullptr) == CFI_SUCCESS);
  std::memcpy(accumulator.OffsetElement<char>(oldLen), from, chars);
  FreeMemory(old);
}

void RTNAME(CharacterAssign)(Descriptor &lhs, const Descriptor &rhs,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  int rank{lhs.rank()};
  RUNTIME_CHECK(terminator, rhs.rank() == 0 || rhs.rank() == rank);
  SubscriptValue ub[maxRank], lhsAt[maxRank], rhsAt[maxRank];
  SubscriptValue elements{1};
  std::size_t lhsBytes{lhs.ElementBytes()};
  std::size_t rhsBytes{rhs.ElementBytes()};
  bool reallocate{lhs.IsAllocatable() &&
      (lhs.raw().base_addr == nullptr || lhsBytes != rhsBytes)};
  for (int j{0}; j < rank; ++j) {
    lhsAt[j] = lhs.GetDimension(j).LowerBound();
    if (rhs.rank() > 0) {
      SubscriptValue lhsExt{lhs.GetDimension(j).Extent()};
      SubscriptValue rhsExt{rhs.GetDimension(j).Extent()};
      ub[j] = lhsAt[j] + rhsExt - 1;
      if (lhsExt != rhsExt) {
        if (lhs.IsAllocatable()) {
          reallocate = true;
        } else {
          terminator.Crash("Character array assignment: operands are not "
                           "conforming on dimension %d (%jd != %jd)",
              j + 1, static_cast<std::intmax_t>(lhsExt),
              static_cast<std::intmax_t>(rhsExt));
        }
      }
      rhsAt[j] = rhs.GetDimension(j).LowerBound();
    } else {
      ub[j] = lhs.GetDimension(j).UpperBound();
    }
    elements *= ub[j] - lhsAt[j] + 1;
  }
  void *old{nullptr};
  if (reallocate) {
    old = lhs.raw().base_addr;
    lhs.set_base_addr(nullptr);
    lhs.raw().elem_len = lhsBytes = rhsBytes;
    if (rhs.rank() > 0) {
      // When the RHS is not scalar, the LHS acquires its bounds.
      for (int j{0}; j < rank; ++j) {
        lhsAt[j] = rhsAt[j];
        ub[j] = rhs.GetDimension(j).UpperBound();
      }
    }
    RUNTIME_CHECK(terminator, lhs.Allocate(lhsAt, ub) == CFI_SUCCESS);
  }
  switch (lhs.raw().type) {
  case CFI_type_char:
    switch (rhs.raw().type) {
    case CFI_type_char:
      for (; elements-- > 0;
           lhs.IncrementSubscripts(lhsAt), rhs.IncrementSubscripts(rhsAt)) {
        CopyAndPad(lhs.Element<char>(lhsAt), rhs.Element<char>(rhsAt), lhsBytes,
            rhsBytes);
      }
      break;
    case CFI_type_char16_t:
      for (; elements-- > 0;
           lhs.IncrementSubscripts(lhsAt), rhs.IncrementSubscripts(rhsAt)) {
        CopyAndPad(lhs.Element<char>(lhsAt), rhs.Element<char16_t>(rhsAt),
            lhsBytes, rhsBytes >> 1);
      }
      break;
    case CFI_type_char32_t:
      for (; elements-- > 0;
           lhs.IncrementSubscripts(lhsAt), rhs.IncrementSubscripts(rhsAt)) {
        CopyAndPad(lhs.Element<char>(lhsAt), rhs.Element<char32_t>(rhsAt),
            lhsBytes, rhsBytes >> 2);
      }
      break;
    default:
      terminator.Crash(
          "RHS of character assignment does not have a character type");
    }
    break;
  case CFI_type_char16_t:
    switch (rhs.raw().type) {
    case CFI_type_char:
      for (; elements-- > 0;
           lhs.IncrementSubscripts(lhsAt), rhs.IncrementSubscripts(rhsAt)) {
        CopyAndPad(lhs.Element<char16_t>(lhsAt), rhs.Element<char>(rhsAt),
            lhsBytes >> 1, rhsBytes);
      }
      break;
    case CFI_type_char16_t:
      for (; elements-- > 0;
           lhs.IncrementSubscripts(lhsAt), rhs.IncrementSubscripts(rhsAt)) {
        CopyAndPad(lhs.Element<char16_t>(lhsAt), rhs.Element<char16_t>(rhsAt),
            lhsBytes >> 1, rhsBytes >> 1);
      }
      break;
    case CFI_type_char32_t:
      for (; elements-- > 0;
           lhs.IncrementSubscripts(lhsAt), rhs.IncrementSubscripts(rhsAt)) {
        CopyAndPad(lhs.Element<char16_t>(lhsAt), rhs.Element<char32_t>(rhsAt),
            lhsBytes >> 1, rhsBytes >> 2);
      }
      break;
    default:
      terminator.Crash(
          "RHS of character assignment does not have a character type");
    }
    break;
  case CFI_type_char32_t:
    switch (rhs.raw().type) {
    case CFI_type_char:
      for (; elements-- > 0;
           lhs.IncrementSubscripts(lhsAt), rhs.IncrementSubscripts(rhsAt)) {
        CopyAndPad(lhs.Element<char32_t>(lhsAt), rhs.Element<char>(rhsAt),
            lhsBytes >> 2, rhsBytes);
      }
      break;
    case CFI_type_char16_t:
      for (; elements-- > 0;
           lhs.IncrementSubscripts(lhsAt), rhs.IncrementSubscripts(rhsAt)) {
        CopyAndPad(lhs.Element<char32_t>(lhsAt), rhs.Element<char16_t>(rhsAt),
            lhsBytes >> 2, rhsBytes >> 1);
      }
      break;
    case CFI_type_char32_t:
      for (; elements-- > 0;
           lhs.IncrementSubscripts(lhsAt), rhs.IncrementSubscripts(rhsAt)) {
        CopyAndPad(lhs.Element<char32_t>(lhsAt), rhs.Element<char32_t>(rhsAt),
            lhsBytes >> 2, rhsBytes >> 2);
      }
      break;
    default:
      terminator.Crash(
          "RHS of character assignment does not have a character type");
    }
    break;
  default:
    terminator.Crash(
        "LHS of character assignment does not have a character type");
  }
  if (reallocate) {
    FreeMemory(old);
  }
}

int RTNAME(CharacterCompareScalar)(const Descriptor &x, const Descriptor &y) {
  Terminator terminator{__FILE__, __LINE__};
  RUNTIME_CHECK(terminator, x.rank() == 0);
  RUNTIME_CHECK(terminator, y.rank() == 0);
  RUNTIME_CHECK(terminator, x.raw().type == y.raw().type);
  switch (x.raw().type) {
  case CFI_type_char:
    return Compare(x.OffsetElement<char>(), y.OffsetElement<char>(),
        x.ElementBytes(), y.ElementBytes());
  case CFI_type_char16_t:
    return Compare(x.OffsetElement<char16_t>(), y.OffsetElement<char16_t>(),
        x.ElementBytes() >> 1, y.ElementBytes() >> 1);
  case CFI_type_char32_t:
    return Compare(x.OffsetElement<char32_t>(), y.OffsetElement<char32_t>(),
        x.ElementBytes() >> 2, y.ElementBytes() >> 2);
  default:
    terminator.Crash("CharacterCompareScalar: bad string type code %d",
        static_cast<int>(x.raw().type));
  }
  return 0;
}

int RTNAME(CharacterCompareScalar1)(
    const char *x, const char *y, std::size_t xChars, std::size_t yChars) {
  return Compare(x, y, xChars, yChars);
}

int RTNAME(CharacterCompareScalar2)(const char16_t *x, const char16_t *y,
    std::size_t xChars, std::size_t yChars) {
  return Compare(x, y, xChars, yChars);
}

int RTNAME(CharacterCompareScalar4)(const char32_t *x, const char32_t *y,
    std::size_t xChars, std::size_t yChars) {
  return Compare(x, y, xChars, yChars);
}

void RTNAME(CharacterCompare)(
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

std::size_t RTNAME(CharacterAppend1)(char *lhs, std::size_t lhsBytes,
    std::size_t offset, const char *rhs, std::size_t rhsBytes) {
  if (auto n{std::min(lhsBytes - offset, rhsBytes)}) {
    std::memcpy(lhs + offset, rhs, n);
    offset += n;
  }
  return offset;
}

void RTNAME(CharacterPad1)(char *lhs, std::size_t bytes, std::size_t offset) {
  if (bytes > offset) {
    std::memset(lhs + offset, ' ', bytes - offset);
  }
}

// Intrinsic functions

void RTNAME(AdjustL)(Descriptor &result, const Descriptor &string,
    const char *sourceFile, int sourceLine) {
  AdjustLR<false>(result, string, sourceFile, sourceLine);
}

void RTNAME(AdjustR)(Descriptor &result, const Descriptor &string,
    const char *sourceFile, int sourceLine) {
  AdjustLR<true>(result, string, sourceFile, sourceLine);
}

std::size_t RTNAME(LenTrim1)(const char *x, std::size_t chars) {
  return LenTrim(x, chars);
}
std::size_t RTNAME(LenTrim2)(const char16_t *x, std::size_t chars) {
  return LenTrim(x, chars);
}
std::size_t RTNAME(LenTrim4)(const char32_t *x, std::size_t chars) {
  return LenTrim(x, chars);
}

void RTNAME(LenTrim)(Descriptor &result, const Descriptor &string, int kind,
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

void RTNAME(Repeat)(Descriptor &result, const Descriptor &string,
    std::size_t ncopies, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  std::size_t origBytes{string.ElementBytes()};
  result.Establish(string.type(), origBytes * ncopies, nullptr, 0);
  if (result.Allocate(nullptr, nullptr) != CFI_SUCCESS) {
    terminator.Crash("REPEAT could not allocate storage for result");
  }
  const char *from{string.OffsetElement()};
  for (char *to{result.OffsetElement()}; ncopies-- > 0; to += origBytes) {
    std::memcpy(to, from, origBytes);
  }
}

void RTNAME(Trim)(Descriptor &result, const Descriptor &string,
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
  result.Establish(string.type(), resultBytes, nullptr, 0);
  RUNTIME_CHECK(terminator, result.Allocate(nullptr, nullptr) == CFI_SUCCESS);
  std::memcpy(result.OffsetElement(), string.OffsetElement(), resultBytes);
}

void RTNAME(CharacterMax)(Descriptor &accumulator, const Descriptor &x,
    const char *sourceFile, int sourceLine) {
  MaxMin<false>(accumulator, x, sourceFile, sourceLine);
}

void RTNAME(CharacterMin)(Descriptor &accumulator, const Descriptor &x,
    const char *sourceFile, int sourceLine) {
  MaxMin<true>(accumulator, x, sourceFile, sourceLine);
}

// TODO: Character MAXVAL/MINVAL
// TODO: Character MAXLOC/MINLOC
}
} // namespace Fortran::runtime
