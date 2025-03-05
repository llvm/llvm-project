// RUN: %clangxx_nsan -O0 -g -mavx %s -o %t
// RUN: env NSAN_OPTIONS=check_nan=true,halt_on_error=0 %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_nsan -O3 -g -mavx %s -o %t
// RUN: env NSAN_OPTIONS=check_nan=true,halt_on_error=0 %run %t 2>&1 | FileCheck %s

#include <cmath>
#include <immintrin.h>
#include <iostream>

void simd_sqrt(const float *input, float *output, size_t size) {
  size_t i = 0;
  for (; i + 7 < size; i += 8) {
    __m256 vec = _mm256_loadu_ps(&input[i]);
    __m256 result = _mm256_sqrt_ps(vec);
    _mm256_storeu_ps(&output[i], result);
  }
  for (; i < size; ++i) {
    output[i] = std::sqrt(input[i]);
    // CHECK: WARNING: NumericalStabilitySanitizer: NaN detected
  }
}

int main() {
  float input[] = {1.0,  2.0,   -3.0,  4.0,   5.0,   6.0,  7.0,
                   8.0,  9.0,   -10.0, 11.0,  12.0,  13.0, 14.0,
                   15.0, -16.0, 17.0,  -18.0, -19.0, -20.0};
  float output[20];
  simd_sqrt(input, output, 20);
  for (int i = 0; i < 20; ++i) {
    std::cout << output[i] << std::endl;
    // CHECK: WARNING: NumericalStabilitySanitizer: NaN detected
  }
  return 0;
}
