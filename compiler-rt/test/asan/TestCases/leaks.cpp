// Test for LeakSanitizer+AddressSanitizer of different sizes.
// REQUIRES: leak-detection
//
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 0 2>&1 | FileCheck %s
// RUN: not %run %t 1 2>&1 | FileCheck %s
// RUN: not %run %t 1000 2>&1 | FileCheck %s
// RUN: not %run %t 1000000 2>&1 | FileCheck %s

#include <cstdlib>
#include <thread>

__attribute__((noopt)) void leak(int n) {
#if defined(__ANDROID__) || defined(__BIONIC__)
  // Bionic does not acutally allocate when n==0, hence
  // there would not be a leak.
  // Re-adjust n so the test can pass.
  if (n == 0)
    n = 1;
#endif

  // Repeat few times to make sure that at least one pointer is
  // not somewhere on the stack.
  for (int i = 0; i < 10; ++i) {
    volatile int *t = new int[n];
    t = nullptr;
  }
}

int main(int argc, char **argv) {
  leak(atoi(argv[1]));
}
// CHECK: LeakSanitizer: detected memory leaks
