// RUN: %clangxx_nsan -O0 -g %s -o %t
// RUN: env NSAN_OPTIONS=check_nan=true,halt_on_error=0 %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_nsan -O3 -g %s -o %t
// RUN: env NSAN_OPTIONS=check_nan=true,halt_on_error=0 %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_nsan -O0 -g %s -o %t
// RUN: env NSAN_OPTIONS=check_nan=true,halt_on_error=1 not %run %t

#include <cmath>
#include <cstdio>

// This function returns a NaN value for triggering the NaN detection.
__attribute__((noinline)) float ReturnNaN(float p, float q) {
  float ret = p / q;
  return ret;
  // CHECK: WARNING: NumericalStabilitySanitizer: NaN detected
}

int main() {
  float val = ReturnNaN(0., 0.);
  printf("%f\n", val);
  // CHECK: WARNING: NumericalStabilitySanitizer: NaN detected
  return 0;
}
