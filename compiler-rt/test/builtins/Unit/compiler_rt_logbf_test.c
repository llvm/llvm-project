// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- compiler_rt_logbf_test.c - Test __compiler_rt_logbf ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file checks __compiler_rt_logbf from the compiler_rt library for
// conformance against libm.
//
//===----------------------------------------------------------------------===//

#define SINGLE_PRECISION
#include <math.h>
#include <stdio.h>
#include "fp_lib.h"

int test__compiler_rt_logbf(fp_t x) {
  fp_t crt_value = __compiler_rt_logbf(x);
  fp_t libm_value = logbf(x);
  // Compare actual rep, e.g. to avoid NaN != the same NaN
  if (toRep(crt_value) != toRep(libm_value)) {
    printf("error: in __compiler_rt_logb(%a [%X]) = %a [%X] !=  %a [%X]\n", x,
           toRep(x), crt_value, toRep(crt_value), libm_value,
           toRep(libm_value));
    return 1;
  }
  return 0;
}

double cases[] = {
    1.e-6, -1.e-6, NAN, -NAN, INFINITY, -INFINITY, -1,
    -0.0,  0.0,    1,   -2,   2,        -0.5,      0.5,
};

int main() {
  const unsigned N = sizeof(cases) / sizeof(cases[0]);
  unsigned i;
  for (i = 0; i < N; ++i) {
    if (test__compiler_rt_logbf(cases[i])) return 1;
  }

  // Test a moving 1 bit, especially to handle denormal values.
  // Test the negation as well.
  rep_t x = signBit;
  while (x) {
    if (test__compiler_rt_logbf(fromRep(x))) return 1;
    if (test__compiler_rt_logbf(fromRep(signBit ^ x))) return 1;
    x >>= 1;
  }
  // Also try a couple moving ones
  x = signBit | (signBit >> 1) | (signBit >> 2);
  while (x) {
    if (test__compiler_rt_logbf(fromRep(x))) return 1;
    if (test__compiler_rt_logbf(fromRep(signBit ^ x))) return 1;
    x >>= 1;
  }

  return 0;
}
