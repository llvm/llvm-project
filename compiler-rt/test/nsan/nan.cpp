// RUN: %clangxx_nsan -O0 -g %s -o %t
// RUN: NSAN_OPTIONS=check_nan=true %run %t 2>&1 | FileCheck %s


#include<cmath>
#include<cstdio>

// This function returns a NaN value for triggering the NaN detection.
__attribute__((noinline))  
float ReturnNaN() {
  float ret = 0.0 / 0.0;
  return ret;
  // CHECK: WARNING: NumericalStabilitySanitizer: NaN detected
}


int main() {
  float val = ReturnNaN();
  printf("val: %f\n", val);
  // CHECK: WARNING: NumericalStabilitySanitizer: NaN detected
  return 0;
}
