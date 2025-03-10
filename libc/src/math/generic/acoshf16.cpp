//===-- Half-precision acoshf16 function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/acoshf16.h"
#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/macros/optimization.h"
#include "src/math/generic/explogxf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, acoshf16, (float16 x)) {
  using FPBits = fputil::FPBits<float16>;
  FPBits xbits(x);

  uint16_t x_u = xbits.uintval();
  uint16_t x_abs = x_u & 0x7fff;

  // acoshf16(x) domain: x >= 1.0
  if (LIBC_UNLIKELY(x_abs < 0x3c00)) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  if (LIBC_UNLIKELY(x == float16(1.0f)))
    return float16(0.0f);

  // acosh(inf) = inf, acosh(NaN) = NaN
  if (LIBC_UNLIKELY(xbits.is_inf_or_nan()))
    return x;

  // Compute in float32 for accuracy
  float xf32 = static_cast<float>(x);
  // sqrt(x^2 - 1)
  float sqrt_term = fputil::sqrt<float>(
      fputil::multiply_add(xf32, xf32, -1.0f));

  // log(x + sqrt(x^2 - 1))
  float result = static_cast<float>(log_eval(xf32 + sqrt_term));

  return fputil::cast<float16>(result);
}

} // namespace LIBC_NAMESPACE_DECL
