// Regression test for https://bugs.llvm.org/show_bug.cgi?id=37523

// RUN: %clangxx_msan -O0 %s -o %t && %run %t
// RUN: %clangxx_msan -O3 %s -o %t && %run %t
// REQUIRES: x86_64-target-arch

#include <assert.h>
#include <emmintrin.h>

int main() {
  volatile int scale = 5;
  volatile auto zz = _mm_div_ps(_mm_set1_ps(255), _mm_set1_ps(scale));
  assert(zz[0] == 51);
  assert(zz[1] == 51);
  assert(zz[2] == 51);
  assert(zz[3] == 51);
}
