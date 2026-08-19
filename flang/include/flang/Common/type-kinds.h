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
#define FORTRAN_INTEGER_KINDS \
  {Fortran::common::KindsEnum::Kind1, Fortran::common::KindsEnum::Kind2, \
      Fortran::common::KindsEnum::Kind4, Fortran::common::KindsEnum::Kind8, \
      Fortran::common::KindsEnum::Kind16}
#define FORTRAN_UNSIGNED_KINDS FORTRAN_INTEGER_KINDS
#define FORTRAN_REAL_KINDS \
  {Fortran::common::KindsEnum::Kind2, Fortran::common::KindsEnum::Kind3, \
      Fortran::common::KindsEnum::Kind4, Fortran::common::KindsEnum::Kind8, \
      Fortran::common::KindsEnum::Kind10, Fortran::common::KindsEnum::Kind16}
#define FORTRAN_LOGICAL_KINDS \
  {Fortran::common::KindsEnum::Kind1, Fortran::common::KindsEnum::Kind2, \
      Fortran::common::KindsEnum::Kind4, Fortran::common::KindsEnum::Kind8}
#define FORTRAN_CHARACTER_KINDS \
  {Fortran::common::KindsEnum::Kind1, Fortran::common::KindsEnum::Kind2, \
      Fortran::common::KindsEnum::Kind4}

namespace Fortran::common {

/// All possible kinds. Used to strongly type Fortran KIND values.
///
/// Cannot use ENUM_CLASS here because it does not support assigned values.
enum class KindsEnum : int {
  InvalidKind = -1,
  NoKind = 0,

  Kind1 = 1,
  Kind2 = 2,
  Kind3 = 3,
  Kind4 = 4,
  Kind8 = 8,
  Kind10 = 10,
  Kind16 = 16,

  // SpecialKinds
  AssumedTypeKind = -3,
  ClassKind = -2,
  TypelessKind = -1,
};

inline constexpr bool operator<(KindsEnum a, KindsEnum b) {
  return static_cast<int>(a) < static_cast<int>(b);
}
inline constexpr bool operator<=(KindsEnum a, KindsEnum b) {
  return static_cast<int>(a) <= static_cast<int>(b);
}
inline constexpr bool operator>(KindsEnum a, KindsEnum b) {
  return static_cast<int>(a) > static_cast<int>(b);
}
inline constexpr bool operator>=(KindsEnum a, KindsEnum b) {
  return static_cast<int>(a) >= static_cast<int>(b);
}

inline constexpr KindsEnum IntegerKinds[] FORTRAN_INTEGER_KINDS;
inline constexpr KindsEnum UnsignedKinds[] FORTRAN_UNSIGNED_KINDS;
inline constexpr KindsEnum RealKinds[] FORTRAN_REAL_KINDS;
inline constexpr KindsEnum LogicalKinds[] FORTRAN_LOGICAL_KINDS;
inline constexpr KindsEnum CharacterKinds[] FORTRAN_CHARACTER_KINDS;

/// Iterable lists of valid kinds for each TypeCategory for use by SearchTypes.
template <TypeCategory CAT> struct KindsByType;
template <> struct KindsByType<TypeCategory::Integer> {
  static inline constexpr KindsEnum kinds[] FORTRAN_INTEGER_KINDS;
};
template <> struct KindsByType<TypeCategory::Unsigned> {
  static inline constexpr KindsEnum kinds[] FORTRAN_UNSIGNED_KINDS;
};
template <> struct KindsByType<TypeCategory::Real> {
  static inline constexpr KindsEnum kinds[] FORTRAN_REAL_KINDS;
};
template <> struct KindsByType<TypeCategory::Complex> {
  static inline constexpr KindsEnum kinds[] FORTRAN_REAL_KINDS;
};
template <> struct KindsByType<TypeCategory::Logical> {
  static inline constexpr KindsEnum kinds[] FORTRAN_LOGICAL_KINDS;
};
template <> struct KindsByType<TypeCategory::Character> {
  static inline constexpr KindsEnum kinds[] FORTRAN_CHARACTER_KINDS;
};
template <> struct KindsByType<TypeCategory::Derived> {
  static inline constexpr KindsEnum kinds[]{KindsEnum::NoKind};
};

static constexpr int maxKind{16};

template <typename T, std::size_t N>
static constexpr bool IsKindInList(const T (&kinds)[N], KindsEnum kind) {
  for (std::size_t i{0}; i < N; ++i) {
    if (kinds[i] == static_cast<T>(kind))
      return true;
  }
  return false;
}

inline constexpr bool IsValidKind(KindsEnum kind) {
  int val{static_cast<int>(kind)};
  return 1 <= val && val <= maxKind;
}

// A predicate that is true when a kind value is a kind that could possibly
// be supported for an intrinsic type category on some target instruction
// set architecture.
static constexpr bool IsValidKindOfIntrinsicType(
    TypeCategory category, KindsEnum kind) {
  switch (category) {
  case TypeCategory::Integer:
  case TypeCategory::Unsigned: {
    constexpr auto &kinds = KindsByType<TypeCategory::Integer>::kinds;
    return IsKindInList(kinds, kind);
  }
  case TypeCategory::Real:
  case TypeCategory::Complex: {
    constexpr auto &kinds = KindsByType<TypeCategory::Real>::kinds;
    return IsKindInList(kinds, kind);
  }
  case TypeCategory::Character: {
    constexpr auto &kinds = KindsByType<TypeCategory::Character>::kinds;
    return IsKindInList(kinds, kind);
  }
  case TypeCategory::Logical: {
    constexpr auto &kinds = KindsByType<TypeCategory::Logical>::kinds;
    return IsKindInList(kinds, kind);
  }
  default:
    return false;
  }
}

static constexpr KindsEnum TypeSizeInBytes(
    TypeCategory category, KindsEnum kind) {
  if (IsValidKindOfIntrinsicType(category, kind)) {
    if (category == TypeCategory::Real || category == TypeCategory::Complex) {
      int precision{PrecisionOfRealKind(static_cast<int>(kind))};
      int bits{BitsForBinaryPrecision(precision)};
      if (bits == 80) { // x87 is stored in 16-byte containers
        bits = 128;
      }
      if (category == TypeCategory::Complex) {
        bits *= 2;
      }
      return static_cast<KindsEnum>(bits >> 3);
    } else {
      return kind;
    }
  } else {
    return KindsEnum::InvalidKind;
  }
}

} // namespace Fortran::common
#endif // FORTRAN_COMMON_TYPE_KINDS_H_
