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
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/math/fmul.h"

namespace LIBC_NAMESPACE_DECL {
  
static LIB_INLINE int64_t range_reduction_sincospi(double x, double &y) {
  double b = 32.0;
  
  //fputil::DoubleDouble prod = fputil::exact_mult(x, b);
  //using DoubleBits = fputil::FPBits<double>;
  //using DoubleStorageType = typename DoubleBits::StorageType;
  
  
  float result = fmul(x, b);
  double k = static_cast<double>(result);
  fputil::DoubleDouble y = fputil::exact_add(res, -res);
  double y = sum.hi;

  // do the exceptions here...

  return static_cast<int64_t>(k);
}
  
LIBC_INLINE void sincospi_eval(double xd, double &sin_k, double &cos_k, double &sin_k, double &cosm1_y) {
    double y;
    int64_t k = range_reduction_sincospi(xd, y);
    sincospi_eval(xd, sin_k, cos_k, sin_y, cosm1_y);
  }
  
LLVM_LIBC_FUNCTION(double, sinpi, (double x)) {
  using FPBits = typename fputil::FPBits<double>;
  FPBits xbits(x);

  uint64_t x_u = xbits.uintval();
  double x_abs = fputil::abs(x);
  double p = 0x1p52; // precision = 52; 2^p
  
  if (x_abs >= p) {
    if (xbits.is_nan())
      return x;
    if  (x.bits.is_inf()) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }
    return FPBits::zero(xbits.sign()).get_val();
  }

  double sin_k, cos_k, sin_y, cosm1_y;
  sincospi_eval(xd, sin_k, cos_k, sin_y, cosm1_y);

  if (LIBC_UNLIKELY(sin_y == 0 && sin_k == 0))
    return FPBits::zero(xbits.sign()).get_val();

  return fputil::multiply_add(sin_y, cos_k, fputil::multiply_add(cosm1_y, sin_k, sin_k)));
}

