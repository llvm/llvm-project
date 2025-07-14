//===-- exp2_fuzz.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc exp2 implementation.
///
//===----------------------------------------------------------------------===//

#include "src/math/exp2.h"
#include "utils/MPFRWrapper/mpfr_inc.h"
#include <math.h>

extern "C" int LLVMFuzzerTestOneInput(double x) {
  // remove NaN and inf
  if (isnan(x) || isinf(x))
    return 0;
  // signed zeros already tested in unit tests
  if (signbit(x) && x == 0.0)
    return 0;
  mpfr_t input;
  mpfr_init2(input, 53);
  mpfr_set_d(input, x, MPFR_RNDN);
  int output = mpfr_exp2(input, input, MPFR_RNDN);
  mpfr_subnormalize(input, output, MPFR_RNDN);
  double to_compare = mpfr_get_d(input, MPFR_RNDN);

  double result = LIBC_NAMESPACE::exp2(x);

  if (result != to_compare)
    __builtin_trap();

  mpfr_clear(input);
  return 0;
}
