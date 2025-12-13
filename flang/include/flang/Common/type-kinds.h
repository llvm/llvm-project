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

namespace Fortran::common {

static constexpr int maxKind{16};

// A predicate that is true when a kind value is a kind that could possibly
// be supported for an intrinsic type category on some target instruction
// set architecture.
static constexpr bool IsValidKindOfIntrinsicType(
    TypeCategory category, std::int64_t kind) {
  switch (category) {
  case TypeCategory::Integer:
  case TypeCategory::Unsigned:
    return kind == 1 || kind == 2 || kind == 4 || kind == 8 || kind == 16;
  case TypeCategory::Real:
  case TypeCategory::Complex:
    return kind == 2 || kind == 3 || kind == 4 || kind == 8 || kind == 10 ||
        kind == 16;
  case TypeCategory::Character:
    return kind == 1 || kind == 2 || kind == 4;
  case TypeCategory::Logical:
    return kind == 1 || kind == 2 || kind == 4 || kind == 8;
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
