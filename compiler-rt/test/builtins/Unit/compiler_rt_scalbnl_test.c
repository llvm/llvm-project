// RUN: %clang_builtins %s %librt -o %t && %run %t

#define QUAD_PRECISION
#include "fp_lib.h"
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>

#if defined(CRT_HAS_TF_MODE)

int test__compiler_rt_scalbnl(const char *mode, fp_t x, int y) {
#if defined(__ve__)
  if (fpclassify(x) == FP_SUBNORMAL)
    return 0;
#  endif
  fp_t crt_value = __compiler_rt_scalbnl(x, y);
  fp_t libm_value = scalbnl(x, y);
  // Consider +/-0 unequal, but disregard the sign/payload of NaN.
  if (toRep(crt_value) != toRep(libm_value) &&
      !(crt_isnan(crt_value) && crt_isnan(libm_value))) {
    // Split expected values into two for printf
    twords x_t, crt_value_t, libm_value_t;
    x_t.all = toRep(x);
    crt_value_t.all = toRep(crt_value);
    libm_value_t.all = toRep(libm_value);
    printf(
        "error: [%s] in __compiler_rt_scalbnl([%llX %llX], %d) = "
        "[%llX %llX] != [%llX %llX]\n",
        mode, (unsigned long long)x_t.s.high, (unsigned long long)x_t.s.low, y,
        (unsigned long long)crt_value_t.s.high,
        (unsigned long long)crt_value_t.s.low,
        (unsigned long long)libm_value_t.s.high,
        (unsigned long long)libm_value_t.s.low);
    return 1;
  }
  return 0;
}

fp_t cases[] = {
    -NAN,
    NAN,
    -INFINITY,
    INFINITY,
    -0.0,
    0.0,
    -1,
    1,
    -2,
    2,
// Since we are comparing the compiler-rt IEEE implementation against libc's
// long double implementation, this test can only succeed if long double
// is an IEEE 128-bit floating point number.
#  if defined(CRT_LDBL_IEEE_F128)
    LDBL_TRUE_MIN,
#  endif
    LDBL_MIN,
    LDBL_MAX,
    -1.001,
    1.001,
    -1.002,
    1.002,
    1.e-6,
    -1.e-6,
    TF_C(0x1.0p-16381),
    TF_C(0x1.0p-16382),
    TF_C(0x1.0p-16383), // subnormal
    TF_C(0x1.0p-16384), // subnormal
};

int iterate_cases(const char *mode) {
  const unsigned N = sizeof(cases) / sizeof(cases[0]);
  unsigned i;
  for (i = 0; i < N; ++i) {
    int j;
    for (j = -5; j <= 5; ++j) {
      printf("%d, %d\n", i, j);
      if (test__compiler_rt_scalbnl(mode, cases[i], j))
        return 1;
    }
    if (test__compiler_rt_scalbnl(mode, cases[i], -100000))
      return 1;
    if (test__compiler_rt_scalbnl(mode, cases[i], 100000))
      return 1;
    if (test__compiler_rt_scalbnl(mode, cases[i], INT_MIN))
      return 1;
    if (test__compiler_rt_scalbnl(mode, cases[i], INT_MAX))
      return 1;
  }
  return 0;
}

int main() {
  if (iterate_cases("default"))
    return 1;

  // Skip rounding mode tests (fesetround) because compiler-rt's quad-precision
  // multiply also ignores the current rounding mode.

  return 0;
}

#else
int main() {
  printf("skipped\n");
  return 0;
}
#endif
