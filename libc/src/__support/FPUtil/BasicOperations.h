//===-- Basic operations on floating point numbers --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_BASICOPERATIONS_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_BASICOPERATIONS_H

#include "FEnvImpl.h"
#include "FPBits.h"

#include "FEnvImpl.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/UInt128.h"
#include "src/__support/common.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE {
namespace fputil {

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T abs(T x) {
  return FPBits<T>(x).abs().get_val();
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T fmin(T x, T y) {
  const FPBits<T> bitx(x), bity(y);

  if (bitx.is_nan()) {
    return y;
  } else if (bity.is_nan()) {
    return x;
  } else if (bitx.sign() != bity.sign()) {
    // To make sure that fmin(+0, -0) == -0 == fmin(-0, +0), whenever x and
    // y has different signs and both are not NaNs, we return the number
    // with negative sign.
    return (bitx.is_neg()) ? x : y;
  } else {
    return (x < y ? x : y);
  }
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T fmax(T x, T y) {
  FPBits<T> bitx(x), bity(y);

  if (bitx.is_nan()) {
    return y;
  } else if (bity.is_nan()) {
    return x;
  } else if (bitx.sign() != bity.sign()) {
    // To make sure that fmax(+0, -0) == +0 == fmax(-0, +0), whenever x and
    // y has different signs and both are not NaNs, we return the number
    // with positive sign.
    return (bitx.is_neg() ? y : x);
  } else {
    return (x > y ? x : y);
  }
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T fmaximum(T x, T y) {
  FPBits<T> bitx(x), bity(y);

  if (bitx.is_nan())
    return x;
  if (bity.is_nan())
    return y;
  if (bitx.sign() != bity.sign())
    return (bitx.is_neg() ? y : x);
  return x > y ? x : y;
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T fminimum(T x, T y) {
  const FPBits<T> bitx(x), bity(y);

  if (bitx.is_nan())
    return x;
  if (bity.is_nan())
    return y;
  if (bitx.sign() != bity.sign())
    return (bitx.is_neg()) ? x : y;
  return x < y ? x : y;
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T fmaximum_num(T x, T y) {
  FPBits<T> bitx(x), bity(y);
  if (bitx.is_signaling_nan() || bity.is_signaling_nan()) {
    fputil::raise_except_if_required(FE_INVALID);
    if (bitx.is_nan() && bity.is_nan())
      return FPBits<T>::quiet_nan().get_val();
  }
  if (bitx.is_nan())
    return y;
  if (bity.is_nan())
    return x;
  if (bitx.sign() != bity.sign())
    return (bitx.is_neg() ? y : x);
  return x > y ? x : y;
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T fminimum_num(T x, T y) {
  FPBits<T> bitx(x), bity(y);
  if (bitx.is_signaling_nan() || bity.is_signaling_nan()) {
    fputil::raise_except_if_required(FE_INVALID);
    if (bitx.is_nan() && bity.is_nan())
      return FPBits<T>::quiet_nan().get_val();
  }
  if (bitx.is_nan())
    return y;
  if (bity.is_nan())
    return x;
  if (bitx.sign() != bity.sign())
    return (bitx.is_neg() ? x : y);
  return x < y ? x : y;
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T fmaximum_mag(T x, T y) {
  FPBits<T> bitx(x), bity(y);

  if (abs(x) > abs(y))
    return x;
  if (abs(y) > abs(x))
    return y;
  return fmaximum(x, y);
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T fminimum_mag(T x, T y) {
  FPBits<T> bitx(x), bity(y);

  if (abs(x) < abs(y))
    return x;
  if (abs(y) < abs(x))
    return y;
  return fminimum(x, y);
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T fmaximum_mag_num(T x, T y) {
  FPBits<T> bitx(x), bity(y);

  if (abs(x) > abs(y))
    return x;
  if (abs(y) > abs(x))
    return y;
  return fmaximum_num(x, y);
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T fminimum_mag_num(T x, T y) {
  FPBits<T> bitx(x), bity(y);

  if (abs(x) < abs(y))
    return x;
  if (abs(y) < abs(x))
    return y;
  return fminimum_num(x, y);
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T fdim(T x, T y) {
  FPBits<T> bitx(x), bity(y);

  if (bitx.is_nan()) {
    return x;
  }

  if (bity.is_nan()) {
    return y;
  }

  return (x > y ? x - y : 0);
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE int canonicalize(T &cx, const T &x) {
  FPBits<T> sx(x);
  if constexpr (get_fp_type<T>() == FPType::X86_Binary80) {
    // All the pseudo and unnormal numbers are not canonical.
    // More precisely :
    // Exponent   |       Significand      | Meaning
    //            | Bits 63-62 | Bits 61-0 |
    // All Ones   |     00     |    Zero   | Pseudo Infinity, Value = Infinty
    // All Ones   |     00     |  Non-Zero | Pseudo NaN, Value = SNaN
    // All Ones   |     01     | Anything  | Pseudo NaN, Value = SNaN
    //            |   Bit 63   | Bits 62-0 |
    // All zeroes |   One      | Anything  | Pseudo Denormal, Value =
    //            |            |           | (−1)**s × m × 2**−16382
    // All Other  |   Zero     | Anything  | Unnormal, Value =
    //  Values    |            |           | (−1)**s × m × 2**−16382
    bool bit63 = sx.get_implicit_bit();
    UInt128 mantissa = sx.get_explicit_mantissa();
    bool bit62 = ((mantissa & (1ULL << 62)) >> 62);
    int exponent = sx.get_biased_exponent();
    if (exponent == 0x7FFF) {
      if (!bit63 && !bit62) {
        if (mantissa == 0)
          cx = FPBits<T>::inf(sx.sign()).get_val();
        else {
          cx = FPBits<T>::quiet_nan(sx.sign(), mantissa).get_val();
          raise_except_if_required(FE_INVALID);
          return 1;
        }
      } else if (!bit63 && bit62) {
        cx = FPBits<T>::quiet_nan(sx.sign(), mantissa).get_val();
        raise_except_if_required(FE_INVALID);
        return 1;
      } else if (LIBC_UNLIKELY(sx.is_signaling_nan())) {
        cx = FPBits<T>::quiet_nan(sx.sign(), sx.get_explicit_mantissa())
                 .get_val();
        raise_except_if_required(FE_INVALID);
        return 1;
      } else
        cx = x;
    } else if (exponent == 0 && bit63)
      cx = FPBits<T>::make_value(mantissa, 0).get_val();
    else if (exponent != 0 && !bit63)
      cx = FPBits<T>::make_value(mantissa, 0).get_val();
    else if (LIBC_UNLIKELY(sx.is_signaling_nan())) {
      cx =
          FPBits<T>::quiet_nan(sx.sign(), sx.get_explicit_mantissa()).get_val();
      raise_except_if_required(FE_INVALID);
      return 1;
    } else
      cx = x;
  } else if (LIBC_UNLIKELY(sx.is_signaling_nan())) {
    cx = FPBits<T>::quiet_nan(sx.sign(), sx.get_explicit_mantissa()).get_val();
    raise_except_if_required(FE_INVALID);
    return 1;
  } else
    cx = x;
  return 0;
}

} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_BASICOPERATIONS_H
