// Test that preloaded runtime works with unsanitized executables.
//
// RUN: %clangxx %s -o %t
// RUN: env LD_PRELOAD=%shared_libasan not %run %t 2>&1 | FileCheck %s

// REQUIRES: asan-dynamic-runtime

// This way of setting LD_PRELOAD does not work with Android test runner.
// REQUIRES: !android

#include <stdlib.h>

extern "C" ssize_t write(int fd, const void *buf, size_t count);

void do_access(void *p) {
  // CHECK: AddressSanitizer: heap-buffer-overflow
  write(1, p, 2);
}

int main(int argc, char **argv) {
  void *p = malloc(1);
  do_access(p);
  return 0;
}
