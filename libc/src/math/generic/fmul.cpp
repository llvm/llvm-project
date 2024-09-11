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

namespace LIBC_NAMESPACE_DECL {
/*
LLVM_LIBC_FUNCTION(float, fmul, (double x, double y)) {
return fputil::generic::mul<float>(x, y);
}
*/
LLVM_LIBC_FUNCTION(float, fmul, (double x, double y)) {
  fputil::DoubleDouble prod = fputil::exact_mult(x, y);
  fputil::FPBits<double> hi_bits(prod.hi), lo_bits(prod.lo);

  if (LIBC_UNLIKELY(hi_bits.is_inf_or_nan() || hi_bits.is_zero())) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return static_cast<float>(prod.hi);
  }
    if (prod.lo == 0.0)
      return static_cast<float>(prod.hi);

    if (lo_bits.sign() != hi_bits.sign()) {
      // Check if sticky bit of hi are all 0
      constexpr uint64_t STICKY_MASK =
          0xFFF'FFFF; // Lower (52 - 23 - 1 = 28 bits)
      uint64_t sticky_bits = (hi_bits.uintval() & STICKY_MASK);
      uint64_t result_bits =
          (sticky_bits == 0) ? (hi_bits.uintval() - 1) : hi_bits.uintval();
      double result = fputil::FPBits<double>(result_bits).get_val();
      return static_cast<float>(result);
  }

  double result = fputil::FPBits<double>(hi_bits.uintval() | 1).get_val();
  return static_cast<float>(result);
}

} // namespace LIBC_NAMESPACE_DECL
