//===-- fsqrt_fuzz.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc fsqrt implementation.
///
//===----------------------------------------------------------------------===//

#include "src/math/fsqrt.h"
#include "utils/MPFRWrapper/mpfr_inc.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <math.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  mpfr_t input;
  mpfr_t out;
  mpfr_init2(input, 53);
  mpfr_init2(out, 128);
  for (size_t i = 0; i < size / sizeof(double); ++i) {
    double x;
    std::memcpy(&x, data, sizeof(double));
    data += sizeof(double);

    // remove NaN, inf, and values outside the accepted range
    if (isnan(x) || isinf(x) || x < 0)
      continue;
    // signed zeros already tested in unit tests
    if (signbit(x) && x == 0.0)
      continue;

    mpfr_set_d(input, x, MPFR_RNDN);
    mpfr_sqrt(out, input, MPFR_RNDN);
    float to_compare = mpfr_get_flt(out, MPFR_RNDN);

    float result = LIBC_NAMESPACE::fsqrt(x);

    if (result != to_compare) {
      std::cout << std::hexfloat << "Failing input: " << x << std::endl;
      std::cout << std::hexfloat << "Failing output: " << result << std::endl;
      std::cout << std::hexfloat << "Expected: " << to_compare << std::endl;
      __builtin_trap();
    }
  }
  mpfr_clear(input);
  mpfr_clear(out);
  return 0;
}
