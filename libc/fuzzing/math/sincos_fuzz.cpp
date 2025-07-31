//===-- sincos_fuzz.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc sincos implementation.
///
//===----------------------------------------------------------------------===//

#include "src/math/sincos.h"
#include "utils/MPFRWrapper/mpfr_inc.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <math.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  mpfr_t input;
  mpfr_t sin_x;
  mpfr_t cos_x;

  mpfr_init2(input, 53);
  mpfr_init2(sin_x, 53);
  mpfr_init2(cos_x, 53);
  for (size_t i = 0; i < size / sizeof(double); ++i) {
    double x;
    std::memcpy(&x, data, sizeof(double));
    data += sizeof(double);

    // remove NaN and inf as preconditions
    if (isnan(x) || isinf(x))
      continue;

    // signed zeros already tested in unit tests
    if (signbit(x) && x == 0.0)
      continue;

    mpfr_set_d(input, x, MPFR_RNDN);
    int output = mpfr_sin_cos(sin_x, cos_x, input, MPFR_RNDN);
    mpfr_subnormalize(sin_x, output, MPFR_RNDN);
    mpfr_subnormalize(cos_x, output, MPFR_RNDN);

    double to_compare_sin = mpfr_get_d(sin_x, MPFR_RNDN);
    double to_compare_cos = mpfr_get_d(cos_x, MPFR_RNDN);

    double sin_res, cos_res;
    LIBC_NAMESPACE::sincos(x, &sin_res, &cos_res);

    if (sin_res != to_compare_sin || cos_res != to_compare_cos) {
      std::cout << std::hexfloat << "Failing input: " << x << std::endl;
      std::cout << std::hexfloat << "Failing sin output: " << sin_res
                << std::endl;
      std::cout << std::hexfloat << "Expected sin: " << to_compare_sin
                << std::endl;
      std::cout << std::hexfloat << "Failing cos output: " << cos_res
                << std::endl;
      std::cout << std::hexfloat << "Expected cos: " << to_compare_cos
                << std::endl;
      __builtin_trap();
    }
  }

  mpfr_clear(input);
  mpfr_clear(sin_x);
  mpfr_clear(cos_x);
  return 0;
}
