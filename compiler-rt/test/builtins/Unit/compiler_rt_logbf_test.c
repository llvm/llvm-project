// RUN: %clang_builtins %s %librt -o %t && %run %t

#define SINGLE_PRECISION
#include "fp_lib.h"
#include "int_math.h"
#include <math.h>
#include <stdio.h>

int test__compiler_rt_logbf(fp_t x) {
#if defined(__ve__)
  if (fpclassify(x) == FP_SUBNORMAL)
    return 0;
#endif
  fp_t crt_value = __compiler_rt_logbf(x);
  fp_t libm_value = logbf(x);
  // `!=` operator on fp_t returns false for NaNs so also check if operands are
  // both NaN. We don't do `toRepr(crt_value) != toRepr(libm_value)` because
  // that treats different representations of NaN as not equivalent.
  if (crt_value != libm_value &&
      !(crt_isnan(crt_value) && crt_isnan(libm_value))) {
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
