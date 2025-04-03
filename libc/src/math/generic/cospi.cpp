//===-- double-precision cospi function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/cospi.h"
#include "sincos_eval.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/math/fmul.h"
#include "src/math/generic/sincosf_utils.h"
#include "src/math/pow.h"

namespace LIBC_NAMESPACE_DECL {

static LIBC_INLINE void sincospi_poly_eval(double k, double y, double &sin_k,
                                           double &cos_k, double &sin_y,

                                           double &cosm1_y) {

  // Q3 = fpminimax(sin(x*pi), 7, [|64...|], [-0.0078125, 0.0078125]);
  sin_k =
      k * fputil::polyeval(k, 0x1.59b6a771a45cbab8p-94, 0x1.921fb54442d1846ap1,
                           -0x1.8633470ba8bd806cp-76, -0x1.4abbce625be56346p2,
                           0x1.d3e01dfd72e97a92p-61, 0x1.466bc67713dbbfp1,
                           -0x1.14c2648595e2ad4p-47, -0x1.32d1cc20b89301fcp-1);

  sin_y =
      y * fputil::polyeval(k, 0x1.59b6a771a45cbab8p-94, 0x1.921fb54442d1846ap1,
                           -0x1.8633470ba8bd806cp-76, -0x1.4abbce625be56346p2,
                           0x1.d3e01dfd72e97a92p-61, 0x1.466bc67713dbbfp1,
                           -0x1.14c2648595e2ad4p-47, -0x1.32d1cc20b89301fcp-1);

  // Q1 = fpminimax(cos(x * pi), 7, [|64...|], [-0.0078125, 0.0078125]);
  cos_k =
      k * fputil::polyeval(k, 0x1p0, 0x1.a5b22c564ee1d862p-84,
                           -0x1.3bd3cc9be45d30e6p2, -0x1.5c2328fefbe60d3ep-66,
                           0x1.03c1f080a6907a6p2, 0x1.569a4d5c5018eecap-51,
                           -0x1.55d1f72455a9848ap0, -0x1.6b18e5f7fc6c39a6p-38);

  cosm1_y =
      y * fputil::polyeval(y, 0x1p0, 0x1.a5b22c564ee1d862p-84,
                           -0x1.3bd3cc9be45d30e6p2, -0x1.5c2328fefbe60d3ep-66,
                           0x1.03c1f080a6907a6p2, 0x1.569a4d5c5018eecap-51,
                           -0x1.55d1f72455a9848ap0, -0x1.6b18e5f7fc6c39a6p-38);
}

LLVM_LIBC_FUNCTION(double, cospi, (double x)) {
  using FPBits = typename fputil::FPBits<double>;
  FPBits xbits(x);

  xbits.set_sign(Sign::POS);

  uint64_t x_abs_ = xbits.uintval();
  double x_abs = fputil::abs(x);
  double p = 0x1p52; // 2^p where p is the precision
  double p2 = 0x1p53;
  double p3 = 1.0;
  if (LIBC_UNLIKELY(x_abs == 0U))
    return p3;

  if (x_abs >= p) {
    if (x_abs < p2)
      return ((x_abs_ & 0x1) ? -p3 : p3);
    if (xbits.is_nan())
      return x;
    if (xbits.is_inf()) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return x + FPBits::quiet_nan().get_val();
    }
    return p3;
  }
  double n = pow(2, -52);
  double k = fputil::nearest_integer(x * n);
  double y = x - k;
  double sin_k, cos_k, sin_y, cosm1_y;

  sincospi_poly_eval(k, y, sin_k, cos_k, sin_y, cosm1_y);

  if (LIBC_UNLIKELY(sin_y == 0 && cos_k == 0))
    return FPBits::zero(xbits.sign()).get_val();

  return fputil::cast<double>(fputil::multiply_add(
      cos_k, cosm1_y, fputil::multiply_add(-sin_k, sin_y, cos_k)));
}
} // namespace LIBC_NAMESPACE_DECL
