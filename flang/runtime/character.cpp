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
#include <algorithm>
#include <cstring>

namespace Fortran::runtime {

template <typename C>
inline int CompareToBlankPadding(const C *x, std::size_t chars) {
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

template <typename C, int shift>
static int Compare(
    const C *x, const C *y, std::size_t xBytes, std::size_t yBytes) {
  auto minBytes{std::min(xBytes, yBytes)};
  if constexpr (shift == 0) {
    // don't use for kind=2 or =4, that would fail on little-endian machines
    int cmp{std::memcmp(x, y, minBytes)};
    if (cmp < 0) {
      return -1;
    }
    if (cmp > 0) {
      return 1;
    }
    if (xBytes == yBytes) {
      return 0;
    }
    x += minBytes;
    y += minBytes;
  } else {
    for (std::size_t n{minBytes >> shift}; n-- > 0; ++x, ++y) {
      if (*x < *y) {
        return -1;
      }
      if (*x > *y) {
        return 1;
      }
    }
  }
  if (int cmp{CompareToBlankPadding(x, (xBytes - minBytes) >> shift)}) {
    return cmp;
  }
  return -CompareToBlankPadding(y, (yBytes - minBytes) >> shift);
}

extern "C" {

void RTNAME(CharacterConcatenate)(Descriptor & /*temp*/,
    const Descriptor & /*operand*/, const char * /*sourceFile*/,
    int /*sourceLine*/) {
  // TODO
}

void RTNAME(CharacterConcatenateScalar)(
    Descriptor & /*temp*/, const char * /*from*/, std::size_t /*byteLength*/) {
  // TODO
}

void RTNAME(CharacterAssign)(Descriptor & /*lhs*/, const Descriptor & /*rhs*/,
    const char * /*sourceFile*/, int /*sourceLine*/) {
  // TODO
}

int RTNAME(CharacterCompareScalar)(const Descriptor &, const Descriptor &) {
  // TODO real soon once there's type codes for character(kind=2 & 4)
  return 0;
}

int RTNAME(CharacterCompareScalar1)(
    const char *x, const char *y, std::size_t xBytes, std::size_t yBytes) {
  return Compare<char, 0>(x, y, xBytes, yBytes);
}

int RTNAME(CharacterCompareScalar2)(const char16_t *x, const char16_t *y,
    std::size_t xBytes, std::size_t yBytes) {
  return Compare<char16_t, 1>(x, y, xBytes, yBytes);
}

int RTNAME(CharacterCompareScalar4)(const char32_t *x, const char32_t *y,
    std::size_t xBytes, std::size_t yBytes) {
  return Compare<char32_t, 2>(x, y, xBytes, yBytes);
}

void RTNAME(CharacterCompare)(
    Descriptor &, const Descriptor &, const Descriptor &) {
  // TODO real soon once there's type codes for character(kind=2 & 4)
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
}
} // namespace Fortran::runtime
