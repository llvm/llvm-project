//===-- Implementation of fmul function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/math/fmul.h"
#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/generic/mul.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include <iostream>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, fmul, (double x, double y)) {

  #ifndef LIBC_TARGET_CPU_HAS_FMA
  return fputil::generic::mul<float>(x, y);
  #else
  fputil::DoubleDouble prod = fputil::exact_mult(x, y);
  float prod_hif = static_cast<float>(prod.hi);
  fputil::FPBits<float> hif_bits(prod_hif);
  using OutFPBits = fputil::FPBits<float>;
  using OutStorageType = typename OutFPBits::StorageType;
  using InFPBits = fputil::FPBits<double>;
  using InStorageType = typename InFPBits::StorageType;

  InFPBits x_bits(x);
  InFPBits y_bits(y);

  Sign result_sign = x_bits.sign() == y_bits.sign() ? Sign::POS : Sign::NEG;

  using DoubleBits = fputil::FPBits<double>;
  using FloatBits = fputil::FPBits<float>;
  double result = prod.hi;
  DoubleBits hi_bits(prod.hi), lo_bits(prod.lo);
  // Check for cases where we need to propagate the sticky bits:
  constexpr uint64_t STICKY_MASK = 0xFFF'FFF; // Lower (52 - 23 - 1 = 28 bits)
  uint64_t sticky_bits = (hi_bits.uintval() & STICKY_MASK);
  if (LIBC_UNLIKELY(sticky_bits == 0)) {
    // Might need to propagate sticky bits:
    if (!(lo_bits.is_inf_or_nan() || lo_bits.is_zero())) {
      // Now prod.lo is nonzero and finite, we need to propagate sticky bits.
      if (lo_bits.sign() != hi_bits.sign())
        result = DoubleBits(hi_bits.uintval() - 1).get_val();
      else
        result = DoubleBits(hi_bits.uintval() | 1).get_val();
    }
  }

  float result_f = static_cast<float>(result);
  FloatBits rf_bits(result_f);
  uint32_t rf_exp = rf_bits.get_biased_exponent();
  if (LIBC_LIKELY(rf_exp > 0 && rf_exp < 2*FloatBits::EXP_BIAS + 1)) {
    return result_f;
  }

  // Now result_f is either inf/nan/zero/denormal.
  if (x_bits.is_nan() || y_bits.is_nan()) {
      if (x_bits.is_signaling_nan() || y_bits.is_signaling_nan())
	fputil::raise_except_if_required(FE_INVALID);

      if (x_bits.is_quiet_nan()) {
        InStorageType x_payload = x_bits.get_mantissa();
        x_payload >>= InFPBits::FRACTION_LEN - OutFPBits::FRACTION_LEN;
        return OutFPBits::quiet_nan(x_bits.sign(),
                                    static_cast<OutStorageType>(x_payload))
            .get_val();
      }

      if (y_bits.is_quiet_nan()) {
        InStorageType y_payload = y_bits.get_mantissa();
        y_payload >>= InFPBits::FRACTION_LEN - OutFPBits::FRACTION_LEN;
        return OutFPBits::quiet_nan(y_bits.sign(),
                                    static_cast<OutStorageType>(y_payload))
            .get_val();
      }

      return OutFPBits::quiet_nan().get_val();
    }

    if (x_bits.is_inf()) {
      if (y_bits.is_zero()) {
	fputil::set_errno_if_required(EDOM);
	fputil::raise_except_if_required(FE_INVALID);
	
        return OutFPBits::quiet_nan().get_val();
      }

      return OutFPBits::inf(result_sign).get_val();
    }

    if (y_bits.is_inf()) {
      if (x_bits.is_zero()) {
	fputil::set_errno_if_required(EDOM);
	fputil::raise_except_if_required(FE_INVALID);
        return OutFPBits::quiet_nan().get_val();
      }

      return OutFPBits::inf(result_sign).get_val();
    }

    // Now either x or y is zero, and the other one is finite.
    if (hif_bits.is_inf()) {
      fputil::set_errno_if_required(ERANGE);
      return OutFPBits::inf(result_sign).get_val();
    }

    if (x_bits.is_zero() || y_bits.is_zero())
      return FloatBits::zero(result_sign).get_val();

    fputil::set_errno_if_required(ERANGE);
    fputil::raise_except_if_required(FE_UNDERFLOW);
    return result_f;
}
}
// namespace LIBC_NAMESPACE_DECL
