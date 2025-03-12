//===-- double-precision sinpi function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sinpi.h"
#include "sincos_eval.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/math/pow.h"
#include "src/math/generic/sincosf_utils.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/math/fmul.h"

namespace LIBC_NAMESPACE_DECL {
 
LLVM_LIBC_FUNCTION(double, sinpi, (double x)) {
  using FPBits = typename fputil::FPBits<double>;
  FPBits xbits(x);

  //uint64_t x_u = xbits.uintval();
  double x_abs = fputil::abs(x);
  double p = 0x1p52; // precision = 52; 2^p

  if (LIBC_UNLIKELY(x_abs == 0U))
    return x;
  
  if (x_abs >= p) {
    if (xbits.is_nan())
      return x;
    if  (xbits.is_inf()) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }
    return FPBits::zero(xbits.sign()).get_val();
  }
  double n = pow(2, -52);
  double k = fputil::nearest_integer(x * n);
  FPBits kbits(x);
  uint64_t ku = kbits.uintval();
  double y = x - k;
  double sin_k, cos_k, sin_y, cosm1_y;
  sincosf_poly_eval(ku, y,  sin_k, cos_k, sin_y, cosm1_y);

  if (LIBC_UNLIKELY(sin_y == 0 && sin_k == 0))
    return FPBits::zero(xbits.sign()).get_val();

  return fputil::multiply_add(sin_y, cos_k, fputil::multiply_add(cosm1_y, sin_k, sin_k));
}
}

