// RUN: %clang_builtins %s %librt -o %t && %run %t

#define DOUBLE_PRECISION
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include "fp_lib.h"

int test__compiler_rt_scalbn(const char *mode, fp_t x, int y) {
  fp_t crt_value = __compiler_rt_scalbn(x, y);
  fp_t libm_value = scalbn(x, y);
  // Consider +/-0 unequal, but disregard the sign/payload of NaN.
  if (toRep(crt_value) != toRep(libm_value) &&
      !(crt_isnan(crt_value) && crt_isnan(libm_value))) {
    printf("error: [%s] in __compiler_rt_scalbn(%a [%llX], %d) = %a [%llX] "
           "!= %a [%llX]\n",
           mode, x, (unsigned long long)toRep(x), y,
           crt_value, (unsigned long long)toRep(crt_value),
           libm_value, (unsigned long long)toRep(libm_value));
    return 1;
  }
  return 0;
}

fp_t cases[] = {
  -NAN, NAN, -INFINITY, INFINITY, -0.0, 0.0, -1, 1, -2, 2,
  DBL_TRUE_MIN, DBL_TRUE_MIN*7, DBL_MIN, DBL_MAX,
  -1.001, 1.001, -1.002, 1.002, 1.e-6, -1.e-6,
  0x1.0p-1021,
  0x1.0p-1022,
  0x1.0p-1023, // subnormal
  0x1.0p-1024, // subnormal
};

int iterate_cases(const char *mode) {
  const unsigned N = sizeof(cases) / sizeof(cases[0]);
  unsigned i;
  for (i = 0; i < N; ++i) {
    int j;
    for (j = -5; j <= 5; ++j) {
      if (test__compiler_rt_scalbn(mode, cases[i], j)) return 1;
    }
    if (test__compiler_rt_scalbn(mode, cases[i], -10000)) return 1;
    if (test__compiler_rt_scalbn(mode, cases[i], 10000)) return 1;
    if (test__compiler_rt_scalbn(mode, cases[i], INT_MIN)) return 1;
    if (test__compiler_rt_scalbn(mode, cases[i], INT_MAX)) return 1;
  }
  return 0;
}

int main() {
  if (iterate_cases("default")) return 1;

  // Rounding mode tests on supported architectures. __compiler_rt_scalbn
  // should have the same rounding behavior as double-precision multiplication.
#if (defined(__arm__) || defined(__aarch64__)) && defined(__ARM_FP) || \
    defined(__i386__) || defined(__x86_64__)
// Skip these tests for MSVC because its scalbn function always behaves as if
// the default rounding mode is set (FE_TONEAREST).
#ifndef _MSC_VER
  fesetround(FE_UPWARD);
  if (iterate_cases("FE_UPWARD")) return 1;

  fesetround(FE_DOWNWARD);
  if (iterate_cases("FE_DOWNWARD")) return 1;

  fesetround(FE_TOWARDZERO);
  if (iterate_cases("FE_TOWARDZERO")) return 1;
#endif

  fesetround(FE_TONEAREST);
  if (iterate_cases("FE_TONEAREST")) return 1;
#endif

  return 0;
}
