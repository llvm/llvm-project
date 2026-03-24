// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ATANPIBF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ATANPIBF16_H

#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/FPUtil/bfloat16.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

LIBC_INLINE bfloat16 atanpibf16(bfloat16 x) {

  using FPBits = fputil::FPBits<bfloat16>;
  FPBits xbits(x);

  uint16_t x_u = xbits.uintval();
  uint16_t x_abs = x_u & 0x7fff;
  bool x_sign = x_u >> 15;
  float sign = (x_sign ? -1.0f : 1.0f);

  // Taylor series for atan-> [x - x^3/3 + x^5/5 - x^7/7 ...]
  // x * [1 - x^2/3 + x^4/5 - x^6/7...] -> x * P(x)
  // atan(x) = x * poly(x^2)
  // atan(x)/x = poly(x^2)
  // atan(x)/(x*pi) = 
  //
  // Degree 14 polynomial of atan(x) generated using Sollya with command :
  // > display = hexadecimal ;
  // > P = fpminimax(atan(x)/(x*pi), [|0, 2, 4, 6, 8, 10, 12, 14|], [|1,SG,SG,SG,SG,SG,SG,SG|], [0, 1]);
  //
  // relative error for the polynomial given by:
  // > dirtyinfnorm(atan(x)/(x*pi) - P(x), [0, 1]);
  // error - 0x1.6e4e44p-25
  // worst case error for it being ~ 
  // satisfying -> error < worst_case
  auto atanpi_eval = [](float x0) {
return fputil::polyeval(
    x0,
    0x1.45f304p-2f,
    -0x1.b29476p-4f,
    0x1.0458d4p-4f,
    -0x1.6d6784p-5f,
    0x1.021eep-5f,
    -0x1.352efap-6f,
    0x1.f724c8p-8f,
    -0x1.84ac1ep-10f
);
 };

  float xf = x;
  float x_sq = xf * xf ;

  // Case 1: |x| <= 1
  if (x_abs <= 0x3f80) {
    // atanpibf16(±0) = ±0
    if (LIBC_UNLIKELY(x_abs == 0))
      return x;
    // atanpibf16(±1) = ±0.25
    if (LIBC_UNLIKELY(x_abs == 0x3f80))
        return fputil::cast<bfloat16>(sign * 0.25f);


    float result = atanpi_eval(x_sq); 
    return fputil::cast<bfloat16>(xf *result );
  }

  // Case 2: |x| > 1 ( But not too large )
    if(x_abs < 0x43a3){
    // atan(x) = sign(x) * (pi/2 - atan(1/|x|))
    // atan(x)/pi = sign(x) * ((pi/2)/pi) - ((atan(1/|x|))/pi))
    // atanpi(x) = sign(x) * ((0.5) - atanpi(1/|x|))
    // Since 1/|x| < 1, we can use the same polynomial.
    float x_inv_sq = 1.0f / x_sq;
    float x_inv = fputil::sqrt<float>(x_inv_sq);

    float result = atanpi_eval(x_inv_sq);
    float atan_inv = (x_inv *  result);
    return fputil::cast<bfloat16>(sign * (0.5 - atan_inv));
  }

  // Case 3: For Large x in bfloat16 the value is close to 0.5 but not exactly 0.5
  if (LIBC_UNLIKELY(x_abs < 0x7F80)) 
    return fputil::cast<bfloat16>(sign * 0x1.fffffep-2f); 

  // Case 4: |x| is ±inf or NaN
  if (xbits.is_nan()) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }
  // atanpibf16( ±inf )/pi = ±((pi/2)/pi) =  ± 1/2
  return fputil::cast<bfloat16>(sign * 0.5f);
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ATANPIBF16_H
