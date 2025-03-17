//===-- Half-precision acoshf16 function ----------------------------------===//
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
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/except_value_utils.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/optimization.h"
#include "src/math/generic/explogxf.h"

namespace LIBC_NAMESPACE_DECL {

static constexpr size_t N_EXCEPTS = 2;
static constexpr fputil::ExceptValues<float16, N_EXCEPTS> ACOSHF16_EXCEPTS{{
    // (input, RZ output, RU offset, RD offset, RN offset)
    // x = 0x1.6ep+1, acoshf16(x) = 0x1.dbp+0 (RZ)
    {0x41B7, 0x3ED8, 1, 0, 0},
    // x = 0x1.c8p+0, acoshf16(x) = 0x1.27cp-1 (RZ)
    {0x3CE4, 0x393E, 1, 0, 1},
}};

LLVM_LIBC_FUNCTION(float16, acoshf16, (float16 x)) {
  using FPBits = fputil::FPBits<float16>;
  FPBits xbits(x);
  uint16_t x_u = xbits.uintval();
  uint16_t x_abs = x_u & 0x7fff;

  // Check for NaN input first.
  if (LIBC_UNLIKELY(xbits.is_nan())) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // Check for infinite inputs.
  if (LIBC_UNLIKELY(xbits.is_inf())) {
    if (xbits.is_neg()) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // Domain error for inputs less than 1.0.
  if (LIBC_UNLIKELY(x_abs < 0x3c00U)) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  // acosh(1.0) exactly equals 0.0
  if (LIBC_UNLIKELY(x_u == 0x3c00U))
    return FPBits::zero().get_val();

  if (auto r = ACOSHF16_EXCEPTS.lookup(xbits.uintval());
      LIBC_UNLIKELY(r.has_value()))
    return r.value();

  float xf = x;
  // High precision polynomial approximation for inputs very close to 1.0
  // Specifically, for inputs within the range [1, 1.25), we employ the
  // following step-by-step Taylor expansion derivation to maintain numerical
  // accuracy:
  //
  // Step-by-step derivation:
  // 1. Define y = acosh(x), thus by definition x = cosh(y).
  //
  // 2. Expand cosh(y) using exponential identities:
  //      cosh(y) = (e^y + e^{-y}) / 2
  //      For small y, let us set y ≈ sqrt(2 * delta), thus:
  //      x ≈ cosh(y) ≈ 1 + delta, for small delta
  //      hence delta = x - 1.
  //
  // 3. Express y explicitly in terms of delta (for small delta):
  //      y = acosh(1 + delta) ≈ sqrt(2 * delta) for very small delta.
  //
  // 4. Use Taylor expansion around delta = 0 to obtain a more accurate
  // polynomial:
  //      acosh(1 + delta) ≈ sqrt(2 * delta) * [1 - delta/12 + 3*delta^2/160 -
  //      5*delta^3/896 + 35*delta^4/18432 + ...] For practical computation and
  //      precision, truncate and fit the polynomial precisely in the range [0,
  //      0.25].
  //
  // 5. The implemented polynomial approximation (coefficients obtained from
  // careful numerical fitting) is:
  //      P(delta) ≈ 1 - 0x1.55551ap-4 * delta + 0x1.33160cp-6 * delta^2 -
  //      0x1.6890f4p-8 * delta^3 + 0x1.8f3a62p-10 * delta^4
  //
  // Since delta = x - 1, and 0 <= delta < 0.25, this approximation achieves
  // high precision and numerical stability.
  if (LIBC_UNLIKELY(xf < 1.25f)) {
    float delta = xf - 1.0f;
    float sqrt_2_delta = fputil::sqrt<float>(2.0 * delta);
    float pe = fputil::polyeval(delta, 0x1p+0f, -0x1.55551ap-4f, 0x1.33160cp-6f,
                                -0x1.6890f4p-8f, 0x1.8f3a62p-10f);
    float approx = sqrt_2_delta * pe;
    return fputil::cast<float16>(approx);
  }

  // acosh(x) = log(x + sqrt(x^2 - 1))
  float sqrt_term = fputil::sqrt<float>(fputil::multiply_add(xf, xf, -1.0f));
  float result = static_cast<float>(log_eval(xf + sqrt_term));

  return fputil::cast<float16>(result);
}

} // namespace LIBC_NAMESPACE_DECL
