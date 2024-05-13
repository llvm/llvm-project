// RUN: %clang_builtins %s %librt -o %t && %run %t

#define QUAD_PRECISION
#include "fp_lib.h"
#include "int_lib.h"
#include <math.h>
#include <stdio.h>

#if defined(CRT_HAS_TF_MODE)

int test__compiler_rt_logbl(fp_t x) {
#  if defined(__ve__)
  if (fpclassify(x) == FP_SUBNORMAL)
    return 0;
#  endif
  fp_t crt_value = __compiler_rt_logbl(x);
  fp_t libm_value = logbl(x);
  // Compare the values, considering all NaNs equivalent, as the spec doesn't
  // specify the NaN signedness/payload.
  if (crt_value != libm_value &&
      !(crt_isnan(crt_value) && crt_isnan(libm_value))) {
    // Split expected values into two for printf
    twords x_t, crt_value_t, libm_value_t;
    x_t.all = toRep(x);
    crt_value_t.all = toRep(crt_value);
    libm_value_t.all = toRep(libm_value);
    printf("error: in __compiler_rt_logbl([%llX %llX]) = [%llX %llX] !=  "
           "[%llX %llX]\n",
           x_t.s.high, x_t.s.low, crt_value_t.s.high, crt_value_t.s.low,
           libm_value_t.s.high, libm_value_t.s.low);
    return 1;
  }
  return 0;
}

fp_t cases[] = {
    1.e-6, -1.e-6, NAN, -NAN, INFINITY, -INFINITY, -1,
    -0.0,  0.0,    1,   -2,   2,        -0.5,      0.5,
};

int main() {
  const unsigned N = sizeof(cases) / sizeof(cases[0]);
  for (unsigned i = 0; i < N; ++i) {
    if (test__compiler_rt_logbl(cases[i]))
      return 1;
  }

  // Test a moving 1 bit, especially to handle denormal values.
  // Test the negation as well.
  // Since we are comparing the compiler-rt IEEE implementation against libc's
  // long double implementation, this test can only succeed if long double
  // is an IEEE 128-bit floating point number (otherwise we will see mismatches
  // once we reach numbers that cannot be precisely represented in long double
  // format).
#  if defined(CRT_LDBL_IEEE_F128)
  rep_t x = signBit;
  int i = 0;
  while (x) {
    if (test__compiler_rt_logbl(fromRep(x)))
      return 1;
    if (test__compiler_rt_logbl(fromRep(signBit ^ x)))
      return 1;
    x >>= 1;
    printf("l1: %d\n", i++);
  }
  // Also try a couple moving ones
  x = signBit | (signBit >> 1) | (signBit >> 2);
  while (x) {
    if (test__compiler_rt_logbl(fromRep(x)))
      return 1;
    if (test__compiler_rt_logbl(fromRep(signBit ^ x)))
      return 1;
    x >>= 1;
    printf("l1: %d\n", i++);
  }
#endif

  return 0;
}
#else
int main() {
  printf("skipped\n");
  return 0;
}
#endif
