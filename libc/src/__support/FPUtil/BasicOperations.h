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
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/__support/uint128.h"

namespace LIBC_NAMESPACE {
namespace fputil {

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T abs(T x) {
  return FPBits<T>(x).abs().get_val();
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T fmin(T x, T y) {
  const FPBits<T> bitx(x), bity(y);

  if (bitx.is_nan())
    return y;
  if (bity.is_nan())
    return x;
  if (bitx.sign() != bity.sign())
    // To make sure that fmin(+0, -0) == -0 == fmin(-0, +0), whenever x and
    // y has different signs and both are not NaNs, we return the number
    // with negative sign.
    return bitx.is_neg() ? x : y;
  return x < y ? x : y;
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T fmax(T x, T y) {
  FPBits<T> bitx(x), bity(y);

  if (bitx.is_nan())
    return y;
  if (bity.is_nan())
    return x;
  if (bitx.sign() != bity.sign())
    // To make sure that fmax(+0, -0) == +0 == fmax(-0, +0), whenever x and
    // y has different signs and both are not NaNs, we return the number
    // with positive sign.
    return bitx.is_neg() ? y : x;
  return x > y ? x : y;
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

// helpers for fmul

uint64_t maxu(uint64_t A, uint64_t B) { return A > B ? A : B; }

uint64_t mul(uint64_t a, uint64_t b) {
  __uint128_t product =
      static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
  return static_cast<uint64_t>(product >> 64);
}

uint64_t mullow(uint64_t a, uint64_t b) {
  __uint128_t product =
      static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
  return static_cast<uint64_t>(product);
}

uint64_t nlz(uint64_t x) {
  uint64_t z = 0;

  if (x == 0)
    return 64;
  if (x <= 0x00000000FFFFFFFF) {
    z = z + 32;
    x = x << 32;
  }
  if (x <= 0x0000FFFFFFFFFFFF) {
    z = z + 16;
    x = x << 16;
  }
  if (x <= 0x00FFFFFFFFFFFFFF) {
    z = z + 8;
    x = x << 8;
  }
  if (x <= 0x0FFFFFFFFFFFFFFF) {
    z = z + 4;
    x = x << 4;
  }
  if (x <= 0x3FFFFFFFFFFFFFFF) {
    z = z + 2;
    x = x << 2;
  }
  if (x <= 0x7FFFFFFFFFFFFFFF) {
    z = z + 1;
  }
  return z;
}

LIBC_INLINE float fmul(double x, double y) {

  auto x_bits = FPBits<double>(x);
  uint64_t x_u = x_bits.uintval();

  auto y_bits = FPBits<double>(y);
  uint64_t y_u = y_bits.uintval();

  uint64_t absx = x_u & 0x7FFFFFFFFFFFFFFF;
  uint64_t absy = y_u & 0x7FFFFFFFFFFFFFFF;

  uint64_t exponent_x = absx >> 52;
  uint64_t exponent_y = absy >> 52;

  // uint64_t x_normal_bit = absx >= 0x10000000000000;
  // uint64_t y_normal_bit = absy >= 0x10000000000000;

  uint64_t mx, my;

  mx = maxu(nlz(absx), 11);

  // lambdax = mx - 11;

  my = maxu(nlz(absy), 11);

  // lambday = my - 11;

  int32_t dm1;
  uint64_t mpx, mpy, highs, lows, b;
  uint64_t g, hight, lowt, morlowt, c, m;
  mpx = (x_u << mx) | 0x8000000000000000;
  mpy = (y_u << my) | 0x8000000000000000;
  highs = mul(mpx, mpy);
  c = highs >= 0x8000000000000000;
  lows = mullow(mpx, mpy);

  lowt = (lows != 0);

  g = (highs >> (38 + c)) & 1;
  hight = (highs << (55 - c)) != 0;

  int32_t exint = static_cast<uint32_t>(exponent_x);
  int32_t eyint = static_cast<uint32_t>(exponent_y);
  int32_t cint = static_cast<uint32_t>(c);
  dm1 = ((exint + eyint) - 1919) + cint;

  if (dm1 >= 255) {
    dm1 = 255;

    m = 0;
  } else if (dm1 <= 0) {
    m = static_cast<uint32_t>((highs >> (39 + c)) >> (1 - dm1));
    dm1 = 0;
    morlowt = m | lowt;
    b = g & (morlowt | hight);
  } else {
    m = static_cast<uint32_t>(highs >> (39 + c));
    morlowt = m | lowt;
    b = g & (morlowt | hight);
  }

  uint32_t sr = static_cast<uint32_t>((x_u ^ y_u) & 0x8000000000000000);

  uint32_t exp16 = sr | (static_cast<uint32_t>(dm1) << 23);
  if (dm1 == 0) {
    uint32_t m2 = static_cast<uint32_t>(m);
    uint32_t result =
        (static_cast<uint32_t>(exp16) + m2) + static_cast<uint32_t>(b);

    float result32 = cpp::bit_cast<float>(result);

    return result32;

  } else {
    constexpr uint32_t FLOAT32_MANTISSA_MASK =
        0b00000000011111111111111111111111;
    uint32_t m2 = static_cast<uint32_t>(m) & FLOAT32_MANTISSA_MASK;

    uint32_t result =
        (static_cast<uint32_t>(exp16) + m2) + static_cast<uint32_t>(b);

    float result16 = cpp::bit_cast<float>(result);
    // std::memcpy(&result16, &result, sizeof(result16));
    // result = *reinterpret_cast<_Float16*>(&result);

    return result16;
  }
}
/*
LIBC_INLINE float fmul(double x, double y){
FPBits<double> bitx(x), bity(y);

long long int product;
long long int p;

if (bitx.is_normal() && bity.is_normal()) {
  product = bitx.get_mantissa() * bity.get_mantissa();
  p = product & ((1ULL << 24) -1);
  sr = bitx.sign() ^ bity.sign();
  return pow(-1, sr) * pow(p, p.biased_exponent());
}
}
*/
LIBC_INLINE float fmull(long double x, long double y) {
  return static_cast<float>(x * y);
}

LIBC_INLINE double dmull(long double x, long double y) {
  return static_cast<double>(x * y);
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE int canonicalize(T &cx, const T &x) {
  FPBits<T> sx(x);
  if constexpr (get_fp_type<T>() == FPType::X86_Binary80) {
    // All the pseudo and unnormal numbers are not canonical.
    // More precisely :
    // Exponent   |       Significand      | Meaning
    //            | Bits 63-62 | Bits 61-0 |
    // All Ones   |     00     |    Zero   | Pseudo Infinity, Value = SNaN
    // All Ones   |     00     |  Non-Zero | Pseudo NaN, Value = SNaN
    // All Ones   |     01     | Anything  | Pseudo NaN, Value = SNaN
    //            |   Bit 63   | Bits 62-0 |
    // All zeroes |   One      | Anything  | Pseudo Denormal, Value =
    //            |            |           | (−1)**s × m × 2**−16382
    // All Other  |   Zero     | Anything  | Unnormal, Value = SNaN
    //  Values    |            |           |
    bool bit63 = sx.get_implicit_bit();
    UInt128 mantissa = sx.get_explicit_mantissa();
    bool bit62 = static_cast<bool>((mantissa & (1ULL << 62)) >> 62);
    int exponent = sx.get_biased_exponent();
    if (exponent == 0x7FFF) {
      if (!bit63 && !bit62) {
        if (mantissa == 0) {
          cx = FPBits<T>::quiet_nan(sx.sign(), mantissa).get_val();
          raise_except_if_required(FE_INVALID);
          return 1;
        }
        cx = FPBits<T>::quiet_nan(sx.sign(), mantissa).get_val();
        raise_except_if_required(FE_INVALID);
        return 1;
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
    else if (exponent != 0 && !bit63) {
      cx = FPBits<T>::quiet_nan(sx.sign(), mantissa).get_val();
      raise_except_if_required(FE_INVALID);
      return 1;
    } else if (LIBC_UNLIKELY(sx.is_signaling_nan())) {
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
