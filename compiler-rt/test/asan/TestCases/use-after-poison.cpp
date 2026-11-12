// Check that __asan_poison_memory_region works.
// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
//
// Check that we can disable it
// RUN: %env_asan_opts=allow_user_poisoning=0 %run %t

#include <stdlib.h>

extern "C" void __asan_poison_memory_region(void *, size_t);

int main(int argc, char **argv) {
  char *x = new char[16];
  x[10] = 0;
  __asan_poison_memory_region(x, 16);
  int res = x[argc * 10];  // BOOOM
  // CHECK: ERROR: AddressSanitizer: use-after-poison
  // CHECK: main{{.*}}use-after-poison.cpp:[[@LINE-2]]
  delete [] x;
  return res;
}
