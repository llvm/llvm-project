//===-- double-precision sinpi function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sinpi.h"
#include "sincos_eval.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
// #include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/generic/mul.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/math/pow.h"
// #include "src/math/generic/range_reduction_double_common.h"
#include "range_reduction_double_nofma.h"
#include "src/__support/FPUtil/multiply_add.h"
#include <iostream>
namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, sinpi, (double x)) {

  // Range reduction:
  // Given x = (k + y) * 1/128
  // find k and y such that
  //   k = round(x * 128)
  //   y = x * 128 - k

  // x = (k + y) * 1/128 and
  // sin(x * pi) = sin((k +y)*pi/128)
  //             = sin(k * pi/128) * cos(y * pi/128) +
  //             = sin(y * pi/128) * cos(k* pi/128)

  using FPBits = typename fputil::FPBits<double>;
  using DoubleDouble = fputil::DoubleDouble;

  double k = fputil::nearest_integer(x * 128);
  FPBits kbits(k);
  FPBits xbits(x);
  uint64_t k_bits = kbits.uintval();

  double fff = 5.0;
  [[maybe_unused]] Float128 ggg = range_reduction_small_f128(fff);

  double y = (x * 128) - k;
  double pi = 3.14 / 128;
  DoubleDouble yy = fputil::exact_mult(y, pi);

  uint64_t abs_u = xbits.uintval();

  uint64_t x_abs = abs_u & 0xFFFFFFFFFFFFFFFF;

  if (LIBC_UNLIKELY(x_abs == 0U))
    return x;

  if (x_abs >= 0x1p52) {
    if (xbits.is_nan())
      return x;
    if (xbits.is_inf()) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return FPBits::zero(xbits.sign()).get_val();
  }

  DoubleDouble sin_y, cos_y;

  [[maybe_unused]] double err = generic::sincos_eval(yy, sin_y, cos_y);
  DoubleDouble sin_k = SIN_K_PI_OVER_128[k_bits & 255];
  DoubleDouble cos_k = SIN_K_PI_OVER_128[(k_bits + 64) & 255];
  double sin_kk = sin_k.hi;
  double cos_kk = cos_k.hi;
  double sin_yy = sin_y.hi;
  double cos_yy = cos_y.hi;

  if (LIBC_UNLIKELY(sin_yy == 0 && sin_kk == 0))
    return FPBits::zero(xbits.sign()).get_val();

  return static_cast<double>(fputil::multiply_add(
      sin_yy, cos_kk, fputil::multiply_add(cos_yy, sin_kk, sin_kk)));
}
} // namespace LIBC_NAMESPACE_DECL
