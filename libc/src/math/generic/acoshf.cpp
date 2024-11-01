//===-- Single-precision acosh function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/acoshf.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/math/generic/common_constants.h"
#include "src/math/generic/explogxf.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, acoshf, (float x)) {
  using FPBits_t = typename fputil::FPBits<float>;
  FPBits_t xbits(x);
  uint32_t x_u = xbits.uintval();

  if (LIBC_UNLIKELY(x <= 1.0f)) {
    if (x == 1.0f)
      return 0.0f;
    // x < 1.
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits_t::build_quiet_nan(0);
  }

  if (LIBC_UNLIKELY(x_u >= 0x4f8ffb03)) {
    // Check for exceptional values.
    uint32_t x_abs = x_u & FPBits_t::FloatProp::EXP_MANT_MASK;
    if (LIBC_UNLIKELY(x_abs >= 0x7f80'0000U)) {
      // x is +inf or NaN.
      return x;
    }

    // Helper functions to set results for exceptional cases.
    auto round_result_slightly_down = [](float r) -> float {
      volatile float tmp = r;
      tmp = tmp - 0x1.0p-25f;
      return tmp;
    };
    auto round_result_slightly_up = [](float r) -> float {
      volatile float tmp = r;
      tmp = tmp + 0x1.0p-25f;
      return tmp;
    };

    switch (x_u) {
    case 0x4f8ffb03: // x = 0x1.1ff606p32f
      return round_result_slightly_up(0x1.6fdd34p4f);
    case 0x5c569e88: // x = 0x1.ad3d1p57f
      return round_result_slightly_up(0x1.45c146p5f);
    case 0x5e68984e: // x = 0x1.d1309cp61f
      return round_result_slightly_up(0x1.5c9442p5f);
    case 0x655890d3: // x = 0x1.b121a6p75f
      return round_result_slightly_down(0x1.a9a3f2p5f);
    case 0x6eb1a8ec: // x = 0x1.6351d8p94f
      return round_result_slightly_down(0x1.08b512p6f);
    case 0x7997f30a: // x = 0x1.2fe614p116f
      return round_result_slightly_up(0x1.451436p6f);
    }
  }

  double x_d = static_cast<double>(x);
  // acosh(x) = log(x + sqrt(x^2 - 1))
  return static_cast<float>(
      log_eval(x_d + fputil::sqrt(fputil::multiply_add(x_d, x_d, -1.0))));
}

} // namespace __llvm_libc
