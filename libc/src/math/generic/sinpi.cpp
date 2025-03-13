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
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/math/fmul.h"
#include "src/math/generic/sincos_eval.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/math/pow.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/math/generic/range_reduction_double_common.h"

namespace LIBC_NAMESPACE_DECL {

const double SIN_K_PI_OVER_128[256] = {
    0x1.92155f7a3667ep-6,  0x1.91f65f10dd814p-5,  0x1.2d52092ce19f6p-4, 
    0x1.917a6bc29b42cp-4,  0x1.f564e56a9730ep-4,  0x1.2c8106e8e613ap-3,
    0x1.5e214448b3fc6p-3,  0x1.8f8b83c69a60bp-3,  0x1.c0b826a7e4f63p-3,
    0x1.f19f97b215f1bp-3,  0x1.111d262b1f677p-2,  0x1.294062ed59f06p-2,
    0x1.4135c94176601p-2,  0x1.58f9a75ab1fddp-2,  0x1.7088530fa459fp-2,
    0x1.87de2a6aea963p-2,  0x1.9ef7943a8ed8ap-2,  0x1.b5d1009e15ccp-2,
    0x1.cc66e9931c45ep-2,  0x1.e2b5d3806f63bp-2,  0x1.f8ba4dbf89abap-2,
    0x1.073879922ffeep-1,  0x1.11eb3541b4b23p-1,  0x1.1c73b39ae68c8p-1,
    0x1.26d054cdd12dfp-1,  0x1.30ff7fce17035p-1,  0x1.3affa292050b9p-1,
    0x1.44cf325091dd6p-1,  0x1.4e6cabbe3e5e9p-1,  0x1.57d69348cecap-1,
    0x1.610b7551d2cdfp-1,  0x1.6a09e667f3bcdp-1,  0x1.72d0837efff96p-1,
    0x1.7b5df226aafafp-1,  0x1.83b0e0bff976ep-1,  0x1.8bc806b151741p-1,
    0x1.93a22499263fbp-1,  0x1.9b3e047f38741p-1,  0x1.a29a7a0462782p-1,
    0x1.a9b66290ea1a3p-1,  0x1.b090a581502p-1,    0x1.b728345196e3ep-1,
    0x1.bd7c0ac6f952ap-1,  0x1.c38b2f180bdb1p-1,  0x1.c954b213411f5p-1,
    0x1.ced7af43cc773p-1,  0x1.d4134d14dc93ap-1,  0x1.d906bcf328d46p-1,
    0x1.ddb13b6ccc23cp-1,  0x1.e212104f686e5p-1,  0x1.e6288ec48e112p-1,
    0x1.e9f4156c62ddap-1,  0x1.ed740e7684963p-1,  0x1.f0a7efb9230d7p-1,
    0x1.f38f3ac64e589p-1,  0x1.f6297cff75cbp-1,   0x1.f8764fa714ba9p-1,
    0x1.fa7557f08a517p-1,  0x1.fc26470e19fd3p-1,  0x1.fd88da3d12526p-1,
    0x1.fe9cdad01883ap-1,  0x1.ff621e3796d7ep-1,  0x1.ffd886084cd0dp-1,
    0x1p0,                 0x1.ffd886084cd0dp-1,  0x1.ff621e3796d7ep-1,
    0x1.fe9cdad01883ap-1,  0x1.fd88da3d12526p-1,  0x1.fc26470e19fd3p-1,
    0x1.fa7557f08a517p-1,  0x1.f8764fa714ba9p-1,  0x1.f6297cff75cbp-1,
    0x1.f38f3ac64e589p-1,  0x1.f0a7efb9230d7p-1,  0x1.ed740e7684963p-1,
    0x1.e9f4156c62ddap-1,  0x1.e6288ec48e112p-1,  0x1.e212104f686e5p-1,
    0x1.ddb13b6ccc23cp-1,  0x1.d906bcf328d46p-1,  0x1.d4134d14dc93ap-1,
    0x1.ced7af43cc773p-1,  0x1.c954b213411f5p-1,  0x1.c38b2f180bdb1p-1,
    0x1.bd7c0ac6f952ap-1,  0x1.b728345196e3ep-1,  0x1.b090a581502p-1,
    0x1.a9b66290ea1a3p-1,  0x1.a29a7a0462782p-1,  0x1.9b3e047f38741p-1,
    0x1.93a22499263fbp-1,  0x1.8bc806b151741p-1,  0x1.83b0e0bff976ep-1,
    0x1.7b5df226aafafp-1,  0x1.72d0837efff96p-1,  0x1.6a09e667f3bcdp-1,
    0x1.610b7551d2cdfp-1,  0x1.57d69348cecap-1,  0x1.4e6cabbe3e5e9p-1,
    0x1.44cf325091dd6p-1,  0x1.3affa292050b9p-1,  0x1.30ff7fce17035p-1,
    0x1.26d054cdd12dfp-1,  0x1.1c73b39ae68c8p-1,  0x1.11eb3541b4b23p-1,
    0x1.073879922ffeep-1,  0x1.f8ba4dbf89abap-2,  0x1.e2b5d3806f63bp-2,
    0x1.cc66e9931c45ep-2,  0x1.b5d1009e15ccp-2,  0x1.9ef7943a8ed8ap-2,
    0x1.87de2a6aea963p-2,  0x1.7088530fa459fp-2,  0x1.58f9a75ab1fddp-2,
    0x1.4135c94176601p-2,  0x1.294062ed59f06p-2,  0x1.111d262b1f677p-2,
    0x1.f19f97b215f1bp-3,  0x1.c0b826a7e4f63p-3,  0x1.8f8b83c69a60bp-3,
    0x1.5e214448b3fc6p-3,  0x1.2c8106e8e613ap-3,  0x1.f564e56a9730ep-4,
    0x1.917a6bc29b42cp-4,  0x1.2d52092ce19f6p-4,  0x1.91f65f10dd814p-5,
    0x1.92155f7a3667ep-6
};

LLVM_LIBC_FUNCTION(double, sinpi, (double x)) {

  // Range reduction:
  // x = (k + y) * 1/128

  // From x find k and y such that
  //   k = round(x * 128)
  //   y = x * 128 - k

  double kd = fputil::nearest_integer(x * 128);
  double yy = fputil::multiply_add<double>(x, 128.0, -kd);

  using FPBits = typename fputil::FPBits<double>;
  
  using DoubleDouble = fputil::DoubleDouble;
  // DoubleDouble y;
  //y.hi = yy;
  //y.lo = 0.0;
  using FPBits = typename fputil::FPBits<double>;
  FPBits xbits(x);
  uint64_t x_u = xbits.uintval();

  uint64_t x_abs = x_u & 0x7fff;

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

  [[maybe_unused]] double err = generic::sincos_eval(y, sin_y, cos_y);
  double sin_k = SIN_K_PI_OVER_128[k & 255];
  double cos_k = SIN_K_PI_OVER_128[(k + 64) & 255];
  
  if (LIBC_UNLIKELY(sin_y.hi == 0 && sin_k == 0))
    return FPBits::zero(xbits.sign()).get_val();
  
  return static_cast<double>(fputil::multiply_add(
      sin_y.hi, cos_k, fputil::multiply_add(cos_y.hi, sin_k, sin_k)));
  
}
} // namespace LIBC_NAMESPACE_DECL
