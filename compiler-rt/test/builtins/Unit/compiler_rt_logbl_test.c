// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- compiler_rt_logbl_test.c - Test __compiler_rt_logbl ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file checks __compiler_rt_logbl from the compiler_rt library for
// conformance against libm.
//
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include <math.h>
#include <stdio.h>
#include "fp_lib.h"
#include "int_lib.h"

#if defined(CRT_HAS_128BIT) && defined(CRT_LDBL_128BIT)

int test__compiler_rt_logbl(fp_t x) {
  fp_t crt_value = __compiler_rt_logbl(x);
  fp_t libm_value = logbl(x);
  // Compare actual rep, e.g. to avoid NaN != the same NaN
  if (toRep(crt_value) != toRep(libm_value)) {
    // Split expected values into two for printf
    twords x_t, crt_value_t, libm_value_t;
    x_t.all = toRep(x);
    crt_value_t.all = toRep(crt_value);
    libm_value_t.all = toRep(libm_value);
    printf(
        "error: in __compiler_rt_logb(%a [%llX %llX]) = %a [%llX %llX] !=  %a "
        "[%llX %llX]\n",
        x, x_t.s.high, x_t.s.low, crt_value, crt_value_t.s.high,
        crt_value_t.s.low, libm_value, libm_value_t.s.high, libm_value_t.s.low);
    return 1;
  }
  return 0;
}

double cases[] = {
    1.e-6, -1.e-6, NAN, -NAN, INFINITY, -INFINITY, -1,
    -0.0,  0.0,    1,   -2,   2,        -0.5,      0.5,
};

#endif

int main() {
#if defined(CRT_HAS_128BIT) && defined(CRT_LDBL_128BIT)
  const unsigned N = sizeof(cases) / sizeof(cases[0]);
  unsigned i;
  for (i = 0; i < N; ++i) {
    if (test__compiler_rt_logbl(cases[i])) return 1;
  }

  // Test a moving 1 bit, especially to handle denormal values.
  // Test the negation as well.
  rep_t x = signBit;
  while (x) {
    if (test__compiler_rt_logbl(fromRep(x))) return 1;
    if (test__compiler_rt_logbl(fromRep(signBit ^ x))) return 1;
    x >>= 1;
  }
  // Also try a couple moving ones
  x = signBit | (signBit >> 1) | (signBit >> 2);
  while (x) {
    if (test__compiler_rt_logbl(fromRep(x))) return 1;
    if (test__compiler_rt_logbl(fromRep(signBit ^ x))) return 1;
    x >>= 1;
  }
#else
  printf("skipped\n");
#endif

  return 0;
}
