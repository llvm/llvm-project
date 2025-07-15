//===-- log_fuzz.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc log implementation.
///
//===----------------------------------------------------------------------===//

#include "src/math/log.h"
#include "utils/MPFRWrapper/mpfr_inc.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <math.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  mpfr_t input;
  mpfr_init2(input, 53);
  for (size_t i = 0; i < size / sizeof(double); ++i) {
    double x;
    std::memcpy(&x, data, sizeof(double));
    data += sizeof(double);

    // remove NaN and inf and values outside accepted range
    if (isnan(x) || isinf(x) || x < 0)
      return 0;
    // signed zeros already tested in unit tests
    if (signbit(x) && x == 0.0)
      return 0;
    mpfr_set_d(input, x, MPFR_RNDN);
    int output = mpfr_log(input, input, MPFR_RNDN);
    mpfr_subnormalize(input, output, MPFR_RNDN);
    double to_compare = mpfr_get_d(input, MPFR_RNDN);

    double result = LIBC_NAMESPACE::log(x);

    if (result != to_compare) {
      std::cout << std::hexfloat << "Failing input: " << x << std::endl;
      std::cout << std::hexfloat << "Failing output: " << result << std::endl;
      std::cout << std::hexfloat << "Expected: " << to_compare << std::endl;
      __builtin_trap();
    }
  }
  mpfr_clear(input);
  return 0;
}
