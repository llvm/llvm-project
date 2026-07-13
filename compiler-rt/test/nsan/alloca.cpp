// RUN: %clangxx_nsan -O0 -g %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_nsan -O3 -g %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <cstddef>

#include "helpers.h"

extern "C" void __nsan_dump_shadow_mem(const char *addr, size_t size_bytes,
                                       size_t bytes_per_line, size_t reserved);

int main() {
  int size = 3 * sizeof(float);
  // Make sure we allocate dynamically: https://godbolt.org/z/T3h998.
  DoNotOptimize(size);
  float *array = reinterpret_cast<float *>(__builtin_alloca(size));
  DoNotOptimize(array);
  array[0] = 1.0;
  array[1] = 2.0;
  // The third float is uninitialized.
  __nsan_dump_shadow_mem((const char *)array, 3 * sizeof(float), 16, 0);
  // CHECK: {{.*}} f0 f1 f2 f3 f0 f1 f2 f3 __ __ __ __ (1.00000000000000000000) (2.00000000000000000000)
  return 0;
}
