//===-- include/flang/Common/type-kinds.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_TYPE_KINDS_H_
#define FORTRAN_COMMON_TYPE_KINDS_H_

#include "Fortran-consts.h"
#include "real.h"
#include <cinttypes>

// Canonical lists of supported Fortran kinds for each intrinsic type.
#define FORTRAN_INTEGER_KINDS {1, 2, 4, 8, 16}
#define FORTRAN_UNSIGNED_KINDS FORTRAN_INTEGER_KINDS
#define FORTRAN_REAL_KINDS {2, 3, 4, 8, 10, 16}
#define FORTRAN_LOGICAL_KINDS {1, 2, 4, 8}
#define FORTRAN_CHARACTER_KINDS {1, 2, 4}

namespace Fortran::common {

static constexpr int maxKind{16};

template <typename T, std::size_t N>
static constexpr bool IsKindInList(const T (&kinds)[N], std::int64_t kind) {
  for (std::size_t i{0}; i < N; ++i) {
    if (kinds[i] == kind)
      return true;
  }
  return false;
}

// A predicate that is true when a kind value is a kind that could possibly
// be supported for an intrinsic type category on some target instruction
// set architecture.
static constexpr bool IsValidKindOfIntrinsicType(
    TypeCategory category, std::int64_t kind) {
  switch (category) {
  case TypeCategory::Integer:
  case TypeCategory::Unsigned: {
    constexpr int kinds[] = FORTRAN_INTEGER_KINDS;
    return IsKindInList(kinds, kind);
  }
  case TypeCategory::Real:
  case TypeCategory::Complex: {
    constexpr int kinds[] = FORTRAN_REAL_KINDS;
    return IsKindInList(kinds, kind);
  }
  case TypeCategory::Character: {
    constexpr int kinds[] = FORTRAN_CHARACTER_KINDS;
    return IsKindInList(kinds, kind);
  }
  case TypeCategory::Logical: {
    constexpr int kinds[] = FORTRAN_LOGICAL_KINDS;
    return IsKindInList(kinds, kind);
  }
  default:
    return false;
  }
}

static constexpr int TypeSizeInBytes(TypeCategory category, std::int64_t kind) {
  if (IsValidKindOfIntrinsicType(category, kind)) {
    if (category == TypeCategory::Real || category == TypeCategory::Complex) {
      int precision{PrecisionOfRealKind(kind)};
      int bits{BitsForBinaryPrecision(precision)};
      if (bits == 80) { // x87 is stored in 16-byte containers
        bits = 128;
      }
      if (category == TypeCategory::Complex) {
        bits *= 2;
      }
      return bits >> 3;
    } else {
      return kind;
    }
  } else {
    return -1;
  }
}

} // namespace Fortran::common
#endif // FORTRAN_COMMON_TYPE_KINDS_H_
