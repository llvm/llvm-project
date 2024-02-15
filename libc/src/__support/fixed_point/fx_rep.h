//===-- Utility class to manipulate fixed point numbers. --*- C++ -*-=========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_FXREP_H
#define LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_FXREP_H

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE, LIBC_INLINE_VAR

#include <stdint.h>

#ifdef LIBC_COMPILER_HAS_FIXED_POINT

namespace LIBC_NAMESPACE::fixed_point {

namespace internal {

template <int Bits> struct Storage {
  static_assert(Bits > 0 && Bits <= 64, "Bits has to be between 1 and 64.");
  using Type = typename cpp::conditional_t<
      (Bits <= 8), uint8_t,
      typename cpp::conditional_t<
          (Bits <= 16 && Bits > 8), uint16_t,
          typename cpp::conditional_t<(Bits <= 32 && Bits > 16), uint32_t,
                                      uint64_t>>>;
};

} // namespace internal

template <typename T> struct FXRep;

template <> struct FXRep<short fract> {
  using Type = short _Fract;

  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = 0;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = SFRACT_FBIT;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN =
      SIGN_LEN + INTEGRAL_LEN + FRACTION_LEN;

  LIBC_INLINE static constexpr Type MIN() { return SFRACT_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return SFRACT_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0.0HR; }
  LIBC_INLINE static constexpr Type EPS() { return SFRACT_EPSILON; }

  using StorageType = typename internal::Storage<TOTAL_LEN>::Type;
  using CompType = cpp::make_signed_t<StorageType>;
};

template <> struct FXRep<unsigned short fract> {
  using Type = unsigned short fract;

  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 0;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = 0;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = USFRACT_FBIT;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN =
      SIGN_LEN + INTEGRAL_LEN + FRACTION_LEN;

  LIBC_INLINE static constexpr Type MIN() { return USFRACT_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return USFRACT_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0.0UHR; }
  LIBC_INLINE static constexpr Type EPS() { return USFRACT_EPSILON; }

  using StorageType = typename internal::Storage<TOTAL_LEN>::Type;
  using CompType = cpp::make_unsigned_t<StorageType>;
};

template <> struct FXRep<fract> {
  using Type = fract;

  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = 0;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = FRACT_FBIT;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN =
      SIGN_LEN + INTEGRAL_LEN + FRACTION_LEN;

  LIBC_INLINE static constexpr Type MIN() { return FRACT_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return FRACT_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0.0R; }
  LIBC_INLINE static constexpr Type EPS() { return FRACT_EPSILON; }

  using StorageType = typename internal::Storage<TOTAL_LEN>::Type;
  using CompType = cpp::make_signed_t<StorageType>;
};

template <> struct FXRep<unsigned fract> {
  using Type = unsigned fract;

  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 0;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = 0;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = UFRACT_FBIT;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN =
      SIGN_LEN + INTEGRAL_LEN + FRACTION_LEN;

  LIBC_INLINE static constexpr Type MIN() { return UFRACT_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return UFRACT_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0.0UR; }
  LIBC_INLINE static constexpr Type EPS() { return UFRACT_EPSILON; }

  using StorageType = typename internal::Storage<TOTAL_LEN>::Type;
  using CompType = cpp::make_unsigned_t<StorageType>;
};

template <> struct FXRep<long fract> {
  using Type = long fract;

  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = 0;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = LFRACT_FBIT;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN =
      SIGN_LEN + INTEGRAL_LEN + FRACTION_LEN;

  LIBC_INLINE static constexpr Type MIN() { return LFRACT_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return LFRACT_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0.0LR; }
  LIBC_INLINE static constexpr Type EPS() { return LFRACT_EPSILON; }

  using StorageType = typename internal::Storage<TOTAL_LEN>::Type;
  using CompType = cpp::make_signed_t<StorageType>;
};

template <> struct FXRep<unsigned long fract> {
  using Type = unsigned long fract;

  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 0;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = 0;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = ULFRACT_FBIT;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN =
      SIGN_LEN + INTEGRAL_LEN + FRACTION_LEN;

  LIBC_INLINE static constexpr Type MIN() { return ULFRACT_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return ULFRACT_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0.0ULR; }
  LIBC_INLINE static constexpr Type EPS() { return ULFRACT_EPSILON; }

  using StorageType = typename internal::Storage<TOTAL_LEN>::Type;
  using CompType = cpp::make_unsigned_t<StorageType>;
};

template <> struct FXRep<short accum> {
  using Type = short accum;

  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = SACCUM_IBIT;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = SACCUM_FBIT;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN =
      SIGN_LEN + INTEGRAL_LEN + FRACTION_LEN;

  LIBC_INLINE static constexpr Type MIN() { return SACCUM_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return SACCUM_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0.0HK; }
  LIBC_INLINE static constexpr Type EPS() { return SACCUM_EPSILON; }

  using StorageType = typename internal::Storage<TOTAL_LEN>::Type;
  using CompType = cpp::make_signed_t<StorageType>;
};

template <> struct FXRep<unsigned short accum> {
  using Type = unsigned short accum;

  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 0;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = USACCUM_IBIT;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = USACCUM_FBIT;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN =
      SIGN_LEN + INTEGRAL_LEN + FRACTION_LEN;

  LIBC_INLINE static constexpr Type MIN() { return USACCUM_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return USACCUM_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0.0UHK; }
  LIBC_INLINE static constexpr Type EPS() { return USACCUM_EPSILON; }

  using StorageType = typename internal::Storage<TOTAL_LEN>::Type;
  using CompType = cpp::make_unsigned_t<StorageType>;
};

template <> struct FXRep<accum> {
  using Type = accum;

  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = ACCUM_IBIT;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = ACCUM_FBIT;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN =
      SIGN_LEN + INTEGRAL_LEN + FRACTION_LEN;

  LIBC_INLINE static constexpr Type MIN() { return ACCUM_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return ACCUM_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0.0K; }
  LIBC_INLINE static constexpr Type EPS() { return ACCUM_EPSILON; }

  using StorageType = typename internal::Storage<TOTAL_LEN>::Type;
  using CompType = cpp::make_signed_t<StorageType>;
};

template <> struct FXRep<unsigned accum> {
  using Type = unsigned accum;

  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 0;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = UACCUM_IBIT;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = UACCUM_FBIT;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN =
      SIGN_LEN + INTEGRAL_LEN + FRACTION_LEN;

  LIBC_INLINE static constexpr Type MIN() { return UACCUM_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return UACCUM_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0.0UK; }
  LIBC_INLINE static constexpr Type EPS() { return UACCUM_EPSILON; }

  using StorageType = typename internal::Storage<TOTAL_LEN>::Type;
  using CompType = cpp::make_unsigned_t<StorageType>;
};

template <> struct FXRep<long accum> {
  using Type = long accum;

  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = LACCUM_IBIT;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = LACCUM_FBIT;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN =
      SIGN_LEN + INTEGRAL_LEN + FRACTION_LEN;

  LIBC_INLINE static constexpr Type MIN() { return LACCUM_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return LACCUM_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0.0LK; }
  LIBC_INLINE static constexpr Type EPS() { return LACCUM_EPSILON; }

  using StorageType = typename internal::Storage<TOTAL_LEN>::Type;
  using CompType = cpp::make_signed_t<StorageType>;
};

template <> struct FXRep<unsigned long accum> {
  using Type = unsigned long accum;

  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 0;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = ULACCUM_IBIT;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = ULACCUM_FBIT;
  LIBC_INLINE_VAR static constexpr int TOTAL_LEN =
      SIGN_LEN + INTEGRAL_LEN + FRACTION_LEN;

  LIBC_INLINE static constexpr Type MIN() { return ULACCUM_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return ULACCUM_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0.0ULK; }
  LIBC_INLINE static constexpr Type EPS() { return ULACCUM_EPSILON; }

  using StorageType = typename internal::Storage<TOTAL_LEN>::Type;
  using CompType = cpp::make_unsigned_t<StorageType>;
};

template <> struct FXRep<short sat fract> : FXRep<short fract> {};
template <> struct FXRep<sat fract> : FXRep<fract> {};
template <> struct FXRep<long sat fract> : FXRep<long fract> {};
template <>
struct FXRep<unsigned short sat fract> : FXRep<unsigned short fract> {};
template <> struct FXRep<unsigned sat fract> : FXRep<unsigned fract> {};
template <>
struct FXRep<unsigned long sat fract> : FXRep<unsigned long fract> {};

template <> struct FXRep<short sat accum> : FXRep<short accum> {};
template <> struct FXRep<sat accum> : FXRep<accum> {};
template <> struct FXRep<long sat accum> : FXRep<long accum> {};
template <>
struct FXRep<unsigned short sat accum> : FXRep<unsigned short accum> {};
template <> struct FXRep<unsigned sat accum> : FXRep<unsigned accum> {};
template <>
struct FXRep<unsigned long sat accum> : FXRep<unsigned long accum> {};

} // namespace LIBC_NAMESPACE::fixed_point

#endif // LIBC_COMPILER_HAS_FIXED_POINT

#endif // LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_FXREP_H
