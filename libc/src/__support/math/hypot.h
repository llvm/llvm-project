//===-- Implementation header for hypot -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_MATH_HYPOT_H
#define LLVM_LIBC_SRC_SUPPORT_MATH_HYPOT_H

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/uint128.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

namespace internal {
// Helper to find the leading one in a mantissa.
template <typename T>
LIBC_INLINE constexpr T find_leading_one(T mant, int &shift_length) {
  shift_length = 0;
  if (mant > 0) {
    shift_length = (sizeof(mant) * 8) - 1 - cpp::countl_zero(mant);
  }
  return static_cast<T>((T(1) << shift_length));
}

// DoubleLength structure mapping
template <typename T> struct DoubleLength;
template <> struct DoubleLength<uint16_t> { using Type = uint32_t; };
template <> struct DoubleLength<uint32_t> { using Type = uint64_t; };
template <> struct DoubleLength<uint64_t> { using Type = UInt128; };

} // namespace internal

// Correctly rounded IEEE 754 HYPOT(x, y) with round to nearest, ties to even.
LIBC_INLINE constexpr double hypot(double x, double y) {
  using FPBits_t = fputil::FPBits<double>;
  using StorageType = FPBits_t::StorageType;
  using DStorageType = internal::DoubleLength<StorageType>::Type;

  FPBits_t x_bits(x);
  FPBits_t y_bits(y);
  
  FPBits_t x_abs = x_bits.abs();
  FPBits_t y_abs = y_bits.abs();

  bool x_abs_larger = x_abs.uintval() >= y_abs.uintval();

  FPBits_t a_bits = x_abs_larger ? x_abs : y_abs;
  FPBits_t b_bits = x_abs_larger ? y_abs : x_abs;

  // 1. Handle Special Cases (Inf / NaN)
  if (LIBC_UNLIKELY(a_bits.is_inf_or_nan())) {
    if (x_abs.is_signaling_nan() || y_abs.is_signaling_nan()) {
       if (!cpp::is_constant_evaluated())
         fputil::raise_except_if_required(FE_INVALID);
       return FPBits_t::quiet_nan().get_val();
    }
    if (x_abs.is_inf() || y_abs.is_inf())
      return FPBits_t::inf().get_val();
    if (x_abs.is_nan())
      return x;
    return y; // y is nan
  }

  uint16_t a_exp = a_bits.get_biased_exponent();
  uint16_t b_exp = b_bits.get_biased_exponent();

  // 2. Trivial Case
  if ((a_exp - b_exp >= FPBits_t::FRACTION_LEN + 2) || (x == 0) || (y == 0)) {
      return x_abs.get_val() + y_abs.get_val();
  }

  // 3. Setup Soft-Float Arithmetic
  uint64_t out_exp = a_exp;
  StorageType a_mant = a_bits.get_mantissa();
  StorageType b_mant = b_bits.get_mantissa();
  
  // FIXED: Initialized variables for constexpr
  DStorageType a_mant_sq = 0;
  DStorageType b_mant_sq = 0;
  bool sticky_bits = false;

  constexpr StorageType ONE = StorageType(1) << (FPBits_t::FRACTION_LEN + 1);

  a_mant <<= 1;
  b_mant <<= 1;

  // FIXED: Initialized variables for constexpr
  StorageType leading_one = 0;
  int y_mant_width = 0;
  
  if (a_exp != 0) {
    leading_one = ONE;
    a_mant |= ONE;
    y_mant_width = FPBits_t::FRACTION_LEN + 1;
  } else {
    leading_one = internal::find_leading_one(a_mant, y_mant_width);
    a_exp = 1;
  }

  if (b_exp != 0)
    b_mant |= ONE;
  else
    b_exp = 1;

  a_mant_sq = static_cast<DStorageType>(a_mant) * a_mant;
  b_mant_sq = static_cast<DStorageType>(b_mant) * b_mant;

  // 4. Align and Add
  uint16_t shift_length = static_cast<uint16_t>(2 * (a_exp - b_exp));
  sticky_bits =
      ((b_mant_sq & ((DStorageType(1) << shift_length) - DStorageType(1))) !=
       DStorageType(0));
  b_mant_sq >>= shift_length;

  DStorageType sum = a_mant_sq + b_mant_sq;

  // 5. Normalize Sum
  if (sum >= (DStorageType(1) << (2 * y_mant_width + 2))) {
    if (leading_one == ONE) {
      sticky_bits = sticky_bits || ((sum & 0x3U) != 0);
      sum >>= 2;
      ++out_exp;
      
      if (out_exp >= FPBits_t::MAX_BIASED_EXPONENT) {
        if (!cpp::is_constant_evaluated()) {
           int round_mode = fputil::quick_get_round();
           if (round_mode == FE_TONEAREST || round_mode == FE_UPWARD)
             return FPBits_t::inf().get_val();
        } else {
           return FPBits_t::inf().get_val();
        }
        return FPBits_t::max_normal().get_val();
      }
    } else {
      leading_one <<= 1;
      ++y_mant_width;
    }
  }

  // 6. Digit-by-Digit Sqrt
  StorageType y_new = leading_one;
  StorageType r = static_cast<StorageType>(sum >> y_mant_width) - leading_one;
  StorageType tail_bits = static_cast<StorageType>(sum) & (leading_one - 1);

  for (StorageType current_bit = leading_one >> 1; current_bit;
       current_bit >>= 1) {
    r = static_cast<StorageType>((r << 1)) +
        ((tail_bits & current_bit) ? 1 : 0);
    StorageType tmp = static_cast<StorageType>((y_new << 1)) +
                      current_bit; 
    if (r >= tmp) {
      r -= tmp;
      y_new += current_bit;
    }
  }

  bool round_bit = y_new & StorageType(1);
  bool lsb = y_new & StorageType(2);

  if (y_new >= ONE) {
    y_new -= ONE;
    if (out_exp == 0) out_exp = 1;
  }

  y_new >>= 1;

  // 7. Rounding
  int round_mode = FE_TONEAREST;
  if (!cpp::is_constant_evaluated()) {
      round_mode = fputil::quick_get_round();
  }

  switch (round_mode) {
  case FE_TONEAREST:
    if (round_bit && (lsb || sticky_bits || (r != 0)))
      ++y_new;
    break;
  case FE_UPWARD:
    if (round_bit || sticky_bits || (r != 0))
      ++y_new;
    break;
  }

  if (y_new >= (ONE >> 1)) {
    y_new -= ONE >> 1;
    ++out_exp;
    if (out_exp >= FPBits_t::MAX_BIASED_EXPONENT) {
      if (round_mode == FE_TONEAREST || round_mode == FE_UPWARD)
        return FPBits_t::inf().get_val();
      return FPBits_t::max_normal().get_val();
    }
  }

  y_new |= static_cast<StorageType>(out_exp) << FPBits_t::FRACTION_LEN;

  if (!cpp::is_constant_evaluated()) {
    if (!(round_bit || sticky_bits || (r != 0)))
        fputil::clear_except_if_required(FE_INEXACT);
  }

  return cpp::bit_cast<double>(y_new);
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SUPPORT_MATH_HYPOT_H