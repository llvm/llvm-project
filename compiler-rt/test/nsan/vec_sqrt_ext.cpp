// RUN: %clangxx_nsan -O0 -g -mavx %s -o %t
// RUN: env NSAN_OPTIONS=check_nan=true,halt_on_error=0 %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_nsan -O3 -g -mavx %s -o %t
// RUN: env NSAN_OPTIONS=check_nan=true,halt_on_error=0 %run %t 2>&1 | FileCheck %s
#include <iostream>
#include <cmath>

typedef float v8sf __attribute__ ((vector_size(32)));

v8sf simd_sqrt(v8sf a) {
  return __builtin_elementwise_sqrt(a);
  // CHECK: WARNING: NumericalStabilitySanitizer: NaN detected
}

int main() {
  v8sf a = {-1.0, -2.0, -3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  a = simd_sqrt(a);

  // This prevents DCE.
  for (size_t i = 0; i < 8; ++i) {
    std::cout << a[i] << std::endl;
    // CHECK: WARNING: NumericalStabilitySanitizer: NaN detected
  }
  return 0;
}
