//===-- Division of IEEE 754 floating-point numbers -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_GENERIC_DIV_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_GENERIC_DIV_H

#include "hdr/fenv_macros.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/dyadic_float.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE::fputil::generic {

template <typename OutType, typename InType>
LIBC_INLINE cpp::enable_if_t<cpp::is_floating_point_v<OutType> &&
                                 cpp::is_floating_point_v<InType> &&
                                 sizeof(OutType) <= sizeof(InType),
                             OutType>
div(InType x, InType y) {
  using OutFPBits = FPBits<OutType>;
  using OutStorageType = typename OutFPBits::StorageType;
  using InFPBits = FPBits<InType>;
  using InStorageType = typename InFPBits::StorageType;
  using DyadicFloat =
      DyadicFloat<cpp::bit_ceil(static_cast<size_t>(InFPBits::FRACTION_LEN))>;
  using DyadicMantissaType = typename DyadicFloat::MantissaType;

  // +1 for the implicit bit.
  constexpr int DYADIC_EXTRA_MANTISSA_LEN =
      DyadicMantissaType::BITS - (InFPBits::FRACTION_LEN + 1);
  // +1 for the extra fractional bit in q.
  constexpr int Q_EXTRA_FRACTION_LEN =
      InFPBits::FRACTION_LEN + 1 - OutFPBits::FRACTION_LEN;

  InFPBits x_bits(x);
  InFPBits y_bits(y);

  Sign result_sign = x_bits.sign() == y_bits.sign() ? Sign::POS : Sign::NEG;

  if (LIBC_UNLIKELY(x_bits.is_inf_or_nan() || y_bits.is_inf_or_nan() ||
                    x_bits.is_zero() || y_bits.is_zero())) {
    if (x_bits.is_nan() || y_bits.is_nan()) {
      if (x_bits.is_signaling_nan() || y_bits.is_signaling_nan())
        raise_except_if_required(FE_INVALID);

      // TODO: Handle NaN payloads.
      return OutFPBits::quiet_nan().get_val();
    }

    if (x_bits.is_inf()) {
      if (y_bits.is_inf()) {
        raise_except_if_required(FE_INVALID);
        return OutFPBits::quiet_nan().get_val();
      }

      return OutFPBits::inf(result_sign).get_val();
    }

    if (y_bits.is_inf())
      return OutFPBits::inf(result_sign).get_val();

    if (y_bits.is_zero()) {
      if (x_bits.is_zero()) {
        raise_except_if_required(FE_INVALID);
        return OutFPBits::quiet_nan().get_val();
      }

      raise_except_if_required(FE_DIVBYZERO);
      return OutFPBits::inf(result_sign).get_val();
    }

    if (x_bits.is_zero())
      return OutFPBits::zero(result_sign).get_val();
  }

  DyadicFloat xd(x);
  DyadicFloat yd(y);

  bool would_q_be_subnormal = xd.mantissa < yd.mantissa;
  int q_exponent = xd.get_unbiased_exponent() - yd.get_unbiased_exponent() -
                   would_q_be_subnormal;

  if (q_exponent + OutFPBits::EXP_BIAS >= OutFPBits::MAX_BIASED_EXPONENT) {
    switch (quick_get_round()) {
    case FE_TONEAREST:
    case FE_UPWARD:
      return OutFPBits::inf().get_val();
    default:
      return OutFPBits::max_normal().get_val();
    }
  }

  if (q_exponent < -OutFPBits::EXP_BIAS - OutFPBits::FRACTION_LEN) {
    switch (quick_get_round()) {
    case FE_UPWARD:
      return OutFPBits::min_subnormal().get_val();
    default:
      return OutFPBits::zero().get_val();
    }
  }

  InStorageType q = 1;
  InStorageType xd_mant_in = static_cast<InStorageType>(
      xd.mantissa >> (DYADIC_EXTRA_MANTISSA_LEN - would_q_be_subnormal));
  InStorageType yd_mant_in =
      static_cast<InStorageType>(yd.mantissa >> DYADIC_EXTRA_MANTISSA_LEN);
  InStorageType r = xd_mant_in - yd_mant_in;

  for (size_t i = 0; i < InFPBits::FRACTION_LEN + 1; i++) {
    q <<= 1;
    InStorageType t = r << 1;
    if (t < yd_mant_in) {
      r = t;
    } else {
      q += 1;
      r = t - yd_mant_in;
    }
  }

  bool round;
  bool sticky;
  OutStorageType result;

  if (q_exponent > -OutFPBits::EXP_BIAS) {
    // Result is normal.

    round = (q & (InStorageType(1) << (Q_EXTRA_FRACTION_LEN - 1))) != 0;
    sticky = (q & ((InStorageType(1) << (Q_EXTRA_FRACTION_LEN - 1)) - 1)) != 0;

    result = OutFPBits::create_value(
                 result_sign,
                 static_cast<OutStorageType>(q_exponent + OutFPBits::EXP_BIAS),
                 static_cast<OutStorageType>((q >> Q_EXTRA_FRACTION_LEN) &
                                             OutFPBits::SIG_MASK))
                 .uintval();

  } else {
    // Result is subnormal.

    // +1 because the leading bit is now part of the fraction.
    int underflow_extra_fraction_len =
        Q_EXTRA_FRACTION_LEN + 1 - q_exponent - OutFPBits::EXP_BIAS;

    InStorageType round_bit_mask = InStorageType(1)
                                   << (underflow_extra_fraction_len - 1);
    round = (q & round_bit_mask) != 0;
    InStorageType sticky_bits_mask = round_bit_mask - 1;
    sticky = (q & sticky_bits_mask) != 0;

    result = OutFPBits::create_value(
                 result_sign, 0,
                 static_cast<OutStorageType>(q >> underflow_extra_fraction_len))
                 .uintval();
  }

  bool lsb = (result & 1) != 0;

  switch (quick_get_round()) {
  case FE_TONEAREST:
    if (round && (lsb || sticky))
      ++result;
    break;
  case FE_UPWARD:
    ++result;
    break;
  default:
    break;
  }

  return cpp::bit_cast<OutType>(result);
}

} // namespace LIBC_NAMESPACE::fputil::generic

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_GENERIC_DIV_H
