// RUN: %clang_builtins %s %librt -o %t && %run %t

#define SINGLE_PRECISION
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include "fp_lib.h"

int test__compiler_rt_scalbnf(const char *mode, fp_t x, int y) {
  fp_t crt_value = __compiler_rt_scalbnf(x, y);
  fp_t libm_value = scalbnf(x, y);
  // Consider +/-0 unequal, but disregard the sign/payload of NaN.
  if (toRep(crt_value) != toRep(libm_value) &&
      !(crt_isnan(crt_value) && crt_isnan(libm_value))) {
    printf("error: [%s] in __compiler_rt_scalbnf(%a [%X], %d) = %a [%X] "
           "!= %a [%X]\n",
           mode, x, toRep(x), y, crt_value, toRep(crt_value),
           libm_value, toRep(libm_value));
    return 1;
  }
  return 0;
}

fp_t cases[] = {
  -NAN, NAN, -INFINITY, INFINITY, -0.0, 0.0, -1, 1, -2, 2,
  FLT_TRUE_MIN, FLT_MIN, FLT_MAX,
  -1.001, 1.001, -1.002, 1.002, 1.e-6, -1.e-6,
  0x1.0p-125,
  0x1.0p-126,
  0x1.0p-127, // subnormal
  0x1.0p-128, // subnormal
};

int iterate_cases(const char *mode) {
  const unsigned N = sizeof(cases) / sizeof(cases[0]);
  unsigned i;
  for (i = 0; i < N; ++i) {
    int j;
    for (j = -5; j <= 5; ++j) {
      if (test__compiler_rt_scalbnf(mode, cases[i], j)) return 1;
    }
    if (test__compiler_rt_scalbnf(mode, cases[i], -1000)) return 1;
    if (test__compiler_rt_scalbnf(mode, cases[i], 1000)) return 1;
    if (test__compiler_rt_scalbnf(mode, cases[i], INT_MIN)) return 1;
    if (test__compiler_rt_scalbnf(mode, cases[i], INT_MAX)) return 1;
  }
  return 0;
}

int main() {
  if (iterate_cases("default")) return 1;

  // Rounding mode tests on supported architectures. __compiler_rt_scalbnf
  // should have the same rounding behavior as single-precision multiplication.
#if (defined(__arm__) || defined(__aarch64__)) && defined(__ARM_FP) || \
    defined(__i386__) || defined(__x86_64__)
  fesetround(FE_UPWARD);
  if (iterate_cases("FE_UPWARD")) return 1;

  fesetround(FE_DOWNWARD);
  if (iterate_cases("FE_DOWNWARD")) return 1;

  fesetround(FE_TOWARDZERO);
  if (iterate_cases("FE_TOWARDZERO")) return 1;

  fesetround(FE_TONEAREST);
  if (iterate_cases("FE_TONEAREST")) return 1;
#endif

  return 0;
}
