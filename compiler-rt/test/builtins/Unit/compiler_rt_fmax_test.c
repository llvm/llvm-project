// RUN: %clang_builtins %s %librt -o %t && %run %t

#define DOUBLE_PRECISION
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include "fp_lib.h"

int test__compiler_rt_fmax(fp_t x, fp_t y) {
  fp_t crt_value = __compiler_rt_fmax(x, y);
  fp_t libm_value = fmax(x, y);
  // Consider +0 and -0 equal, and also disregard the sign/payload of two NaNs.
  if (crt_value != libm_value &&
      !(crt_isnan(crt_value) && crt_isnan(libm_value))) {
    printf("error: in __compiler_rt_fmax(%a [%llX], %a [%llX]) = %a [%llX] "
           "!= %a [%llX]\n",
           x, (unsigned long long)toRep(x),
           y, (unsigned long long)toRep(y),
           crt_value, (unsigned long long)toRep(crt_value),
           libm_value, (unsigned long long)toRep(libm_value));
    return 1;
  }
  return 0;
}

fp_t cases[] = {
  -NAN, NAN, -INFINITY, INFINITY, -0.0, 0.0, -1, 1, -2, 2,
  -0x1.0p-1023, 0x1.0p-1023, -0x1.0p-1024, 0x1.0p-1024, // subnormals
  -1.001, 1.001, -1.002, 1.002,
};

int main() {
  const unsigned N = sizeof(cases) / sizeof(cases[0]);
  unsigned i, j;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      if (test__compiler_rt_fmax(cases[i], cases[j])) return 1;
    }
  }
  return 0;
}
