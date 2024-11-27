// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=allocator_may_return_null=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: %env_asan_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK2

// CHECK1: exceeds maximum supported size
// CHECK1: ABORT

// CHECK2: Success

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

int main() {
  // Attempt to allocate an excessive amount of memory, which should
  // terminate the program unless `allocator_may_return_null` is set.
  size_t max = std::numeric_limits<size_t>::max();

  free(malloc(max));
  printf("Success");
  return 0;
}
