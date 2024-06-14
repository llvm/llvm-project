//===-- Square root of IEEE 754 floating point numbers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_GENERIC_SQRT_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_GENERIC_SQRT_H

#include "sqrt_80_bit_long_double.h"
#include "src/__support/CPP/bit.h" // countl_zero
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/common.h"
#include "src/__support/uint128.h"

#include "hdr/fenv_macros.h"

namespace LIBC_NAMESPACE {
namespace fputil {

namespace internal {

template <typename T> struct SpecialLongDouble {
  static constexpr bool VALUE = false;
};

#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
template <> struct SpecialLongDouble<long double> {
  static constexpr bool VALUE = true;
};
#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

template <typename T>
LIBC_INLINE void normalize(int &exponent,
                           typename FPBits<T>::StorageType &mantissa) {
  const int shift =
      cpp::countl_zero(mantissa) -
      (8 * static_cast<int>(sizeof(mantissa)) - 1 - FPBits<T>::FRACTION_LEN);
  exponent -= shift;
  mantissa <<= shift;
}

#ifdef LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64
template <>
LIBC_INLINE void normalize<long double>(int &exponent, uint64_t &mantissa) {
  normalize<double>(exponent, mantissa);
}
#elif !defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
template <>
LIBC_INLINE void normalize<long double>(int &exponent, UInt128 &mantissa) {
  const uint64_t hi_bits = static_cast<uint64_t>(mantissa >> 64);
  const int shift =
      hi_bits ? (cpp::countl_zero(hi_bits) - 15)
              : (cpp::countl_zero(static_cast<uint64_t>(mantissa)) + 49);
  exponent -= shift;
  mantissa <<= shift;
}
#endif

} // namespace internal

// Correctly rounded IEEE 754 SQRT for all rounding modes.
// Shift-and-add algorithm.
template <typename OutType, typename InType>
LIBC_INLINE cpp::enable_if_t<cpp::is_floating_point_v<OutType> &&
                                 cpp::is_floating_point_v<InType> &&
                                 sizeof(OutType) <= sizeof(InType),
                             OutType>
sqrt(InType x) {
  if constexpr (internal::SpecialLongDouble<OutType>::VALUE &&
                internal::SpecialLongDouble<InType>::VALUE) {
    // Special 80-bit long double.
    return x86::sqrt(x);
  } else {
    // IEEE floating points formats.
    using OutFPBits = typename fputil::FPBits<OutType>;
    using OutStorageType = typename OutFPBits::StorageType;
    using InFPBits = typename fputil::FPBits<InType>;
    using InStorageType = typename InFPBits::StorageType;
    constexpr InStorageType ONE = InStorageType(1) << InFPBits::FRACTION_LEN;
    constexpr auto FLT_NAN = OutFPBits::quiet_nan().get_val();
    constexpr int EXTRA_FRACTION_LEN =
        InFPBits::FRACTION_LEN - OutFPBits::FRACTION_LEN;
    constexpr InStorageType EXTRA_FRACTION_MASK =
        (InStorageType(1) << EXTRA_FRACTION_LEN) - 1;

    InFPBits bits(x);

    if (bits == InFPBits::inf(Sign::POS) || bits.is_zero() || bits.is_nan()) {
      // sqrt(+Inf) = +Inf
      // sqrt(+0) = +0
      // sqrt(-0) = -0
      // sqrt(NaN) = NaN
      // sqrt(-NaN) = -NaN
      return static_cast<OutType>(x);
    } else if (bits.is_neg()) {
      // sqrt(-Inf) = NaN
      // sqrt(-x) = NaN
      return FLT_NAN;
    } else {
      int x_exp = bits.get_exponent();
      InStorageType x_mant = bits.get_mantissa();

      // Step 1a: Normalize denormal input and append hidden bit to the mantissa
      if (bits.is_subnormal()) {
        ++x_exp; // let x_exp be the correct exponent of ONE bit.
        internal::normalize<InType>(x_exp, x_mant);
      } else {
        x_mant |= ONE;
      }

      // Step 1b: Make sure the exponent is even.
      if (x_exp & 1) {
        --x_exp;
        x_mant <<= 1;
      }

      // After step 1b, x = 2^(x_exp) * x_mant, where x_exp is even, and
      // 1 <= x_mant < 4.  So sqrt(x) = 2^(x_exp / 2) * y, with 1 <= y < 2.
      // Notice that the output of sqrt is always in the normal range.
      // To perform shift-and-add algorithm to find y, let denote:
      //   y(n) = 1.y_1 y_2 ... y_n, we can define the nth residue to be:
      //   r(n) = 2^n ( x_mant - y(n)^2 ).
      // That leads to the following recurrence formula:
      //   r(n) = 2*r(n-1) - y_n*[ 2*y(n-1) + 2^(-n-1) ]
      // with the initial conditions: y(0) = 1, and r(0) = x - 1.
      // So the nth digit y_n of the mantissa of sqrt(x) can be found by:
      //   y_n = 1 if 2*r(n-1) >= 2*y(n - 1) + 2^(-n-1)
      //         0 otherwise.
      InStorageType y = ONE;
      InStorageType r = x_mant - ONE;

      for (InStorageType current_bit = ONE >> 1; current_bit;
           current_bit >>= 1) {
        r <<= 1;
        InStorageType tmp = (y << 1) + current_bit; // 2*y(n - 1) + 2^(-n-1)
        if (r >= tmp) {
          r -= tmp;
          y += current_bit;
        }
      }

      // We compute one more iteration in order to round correctly.
      bool lsb = (y & (InStorageType(1) << EXTRA_FRACTION_LEN)) !=
                 0;    // Least significant bit
      bool rb = false; // Round bit
      r <<= 2;
      InStorageType tmp = (y << 2) + 1;
      if (r >= tmp) {
        r -= tmp;
        rb = true;
      }

      bool sticky = false;

      if constexpr (EXTRA_FRACTION_LEN > 0) {
        sticky = rb || (y & EXTRA_FRACTION_MASK) != 0;
        rb = (y & (InStorageType(1) << (EXTRA_FRACTION_LEN - 1))) != 0;
      }

      // Remove hidden bit and append the exponent field.
      x_exp = ((x_exp >> 1) + OutFPBits::EXP_BIAS);

      OutStorageType y_out = static_cast<OutStorageType>(
          ((y - ONE) >> EXTRA_FRACTION_LEN) |
          (static_cast<OutStorageType>(x_exp) << OutFPBits::FRACTION_LEN));

      if constexpr (EXTRA_FRACTION_LEN > 0) {
        if (x_exp >= OutFPBits::MAX_BIASED_EXPONENT) {
          switch (quick_get_round()) {
          case FE_TONEAREST:
          case FE_UPWARD:
            return OutFPBits::inf().get_val();
          default:
            return OutFPBits::max_normal().get_val();
          }
        }

        if (x_exp <
            -OutFPBits::EXP_BIAS - OutFPBits::SIG_LEN + EXTRA_FRACTION_LEN) {
          switch (quick_get_round()) {
          case FE_UPWARD:
            return OutFPBits::min_subnormal().get_val();
          default:
            return OutType(0.0);
          }
        }

        if (x_exp <= 0) {
          int underflow_extra_fraction_len = EXTRA_FRACTION_LEN - x_exp + 1;
          InStorageType underflow_extra_fraction_mask =
              (InStorageType(1) << underflow_extra_fraction_len) - 1;

          rb = (y & (InStorageType(1) << (underflow_extra_fraction_len - 1))) !=
               0;
          OutStorageType subnormal_mant =
              static_cast<OutStorageType>(y >> underflow_extra_fraction_len);
          lsb = (subnormal_mant & 1) != 0;
          sticky = sticky || (y & underflow_extra_fraction_mask) != 0;

          switch (quick_get_round()) {
          case FE_TONEAREST:
            if (rb && (lsb || sticky))
              ++subnormal_mant;
            break;
          case FE_UPWARD:
            if (rb || sticky)
              ++subnormal_mant;
            break;
          }

          return cpp::bit_cast<OutType>(subnormal_mant);
        }
      }

      switch (quick_get_round()) {
      case FE_TONEAREST:
        // Round to nearest, ties to even
        if (rb && (lsb || (r != 0)))
          ++y_out;
        break;
      case FE_UPWARD:
        if (rb || (r != 0) || sticky)
          ++y_out;
        break;
      }

      return cpp::bit_cast<OutType>(y_out);
    }
  }
}

} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_GENERIC_SQRT_H
