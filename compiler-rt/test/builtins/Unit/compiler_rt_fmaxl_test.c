// RUN: %clang_builtins %s %librt -o %t && %run %t

#define QUAD_PRECISION
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include "fp_lib.h"

#if defined(CRT_HAS_TF_MODE)

int test__compiler_rt_fmaxl(fp_t x, fp_t y) {
  fp_t crt_value = __compiler_rt_fmaxl(x, y);
  fp_t libm_value = fmaxl(x, y);
  // Consider +0 and -0 equal, and also disregard the sign/payload of two NaNs.
  if (crt_value != libm_value &&
      !(crt_isnan(crt_value) && crt_isnan(libm_value))) {
    // Split expected values into two for printf
    twords x_t, y_t, crt_value_t, libm_value_t;
    x_t.all = toRep(x);
    y_t.all = toRep(y);
    crt_value_t.all = toRep(crt_value);
    libm_value_t.all = toRep(libm_value);
    printf(
        "error: in __compiler_rt_fmaxl([%llX %llX], [%llX %llX]) = "
        "[%llX %llX] != [%llX %llX]\n",
        (unsigned long long)x_t.s.high, (unsigned long long)x_t.s.low,
        (unsigned long long)y_t.s.high, (unsigned long long)y_t.s.low,
        (unsigned long long)crt_value_t.s.high,
        (unsigned long long)crt_value_t.s.low,
        (unsigned long long)libm_value_t.s.high,
        (unsigned long long)libm_value_t.s.low);
    return 1;
  }
  return 0;
}

fp_t cases[] = {
  -NAN, NAN, -INFINITY, INFINITY, -0.0, 0.0, -1, 1, -2, 2,
  -0x1.0p-16383L, 0x1.0p-16383L, -0x1.0p-16384L, 0x1.0p-16384L, // subnormals
  -1.001, 1.001, -1.002, 1.002,
};

#endif

int main() {
#if defined(CRT_HAS_TF_MODE)
  const unsigned N = sizeof(cases) / sizeof(cases[0]);
  unsigned i, j;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      if (test__compiler_rt_fmaxl(cases[i], cases[j])) return 1;
    }
  }
#else
  printf("skipped\n");
#endif
  return 0;
}
