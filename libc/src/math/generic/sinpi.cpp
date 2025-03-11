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
  double res = static_cast<double>(result);
  fputil::DoubleDouble sum = fputil::exact_add(res, -res);
  double sum_result = sum.hi;

  // do the exceptions here...

  return static_cast<int64_t>(sum_result);
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
  uint64_t x_abs = x_u & 0x7fff'ffffU;

  if (LIBC_UNLIKELY(x_abs >= 0x7c00)) {
    // If value is equal to infinity
    if (x_abs == 0x7c00) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
    }
    return x + FPBits::quiet_nan().get_val();
  } else {
    return FPBits::zero(xbits.sign()).get_val();
  }

  if (LIBC_UNLIKELY(x_abs <= 0x3d80'0000U)) {
    if (LIBC_UNLIKELY(x_abs < 0x33CD'01D7U)) {
      if (LIBC_UNLIKELY(x_abs == 0U)) {
	return x;
    }
      double  xdpi = xd * 0x1.921fb5444d18p1;
      return xdpi;
      
    }
    double xsq = xd * xd;
    // todo: generate a new polynomial using double precision
    double result = fputil::polyeval(xsq,0x1.921fb54442d183f07b2385653d8p1, -0x1.4abbce625bd95cdc955aeed9abcp2, 0x1.466bc6769ddfdb085486c0ff3ep1, -0x1.32d2c4a48bfd71fa9cdf60a0e4p-1,  0x1.502cbd2c72e3168ff209bc7656cp-4);

  return (xd * result);
  }

  double sin_k, cos_k, sin_y, cosm1_y;
  sincospi_eval(xd, sin_k, cos_k, sin_y, cosm1_y);

  if (LIBC_UNLIKELY(sin_y == 0 && sin_k == 0))
    return FPBits::zero(xbits.sign()).get_val();

  return fputil::multiply_add(sin_y, cos_k, fputil::multiply_add(cosm1_y, sin_k, sin_k)));
}
}
