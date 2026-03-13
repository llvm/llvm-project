// RUN: rm -rf %t; mkdir %t
// RUN: %clang_hwasan -Wl,--build-id -g %s -o %t/hwasan_overflow
// RUN: %env_hwasan_opts=symbolize=0 not %run %t/hwasan_overflow 16 2>&1 | hwasan_symbolize --symbols %t --index | FileCheck %s --check-prefixes=CHECK,AFTER0
// RUN: %env_hwasan_opts=symbolize=0 not %run %t/hwasan_overflow 17 2>&1 | hwasan_symbolize --symbols %t --index | FileCheck %s --check-prefixes=CHECK,AFTER1
// RUN: %env_hwasan_opts=symbolize=0 not %run %t/hwasan_overflow -1 2>&1 | hwasan_symbolize --symbols %t --index | FileCheck %s --check-prefixes=CHECK,BEFORE1
// RUN: %env_hwasan_opts=symbolize=0 not %run %t/hwasan_overflow -17 2>&1 | hwasan_symbolize --symbols %t --index | FileCheck %s --check-prefixes=CHECK,BEFORE17
// RUN: %env_hwasan_opts=symbolize=0 not %run %t/hwasan_overflow 1016 2>&1 | hwasan_symbolize --symbols %t --index | FileCheck %s --check-prefixes=CHECK,AFTER1000
// RUN: %env_hwasan_opts=symbolize=0 not %run %t/hwasan_overflow -1000 2>&1 | hwasan_symbolize --symbols %t --index | FileCheck %s --check-prefixes=CHECK,BEFORE1000
// RUN: %env_hwasan_opts=symbolize=0 not %run %t/hwasan_overflow 1000000 2>&1 | hwasan_symbolize --symbols %t --index | FileCheck %s --check-prefixes=TOOFAR

// REQUIRES: aarch64-target-arch

#include <stdlib.h>

static volatile char sink;

int main(int argc, char **argv) {
  volatile char x[16];
  sink = x[atoi(argv[1])];
  // TOOFAR-NOT: Potentially referenced stack object:
  // CHECK: Potentially referenced stack object:
  // AFTER0:   0 bytes after a variable "x" in stack frame of function "main"
  // AFTER1:   1 bytes after a variable "x" in stack frame of function "main"
  // BEFORE1:   1 bytes before a variable "x" in stack frame of function "main"
  // BEFORE17:   17 bytes before a variable "x" in stack frame of function "main"
  // BEFORE1000:   1000 bytes before a variable "x" in stack frame of function "main"
  // AFTER1000:   1000 bytes after a variable "x" in stack frame of function "main"
  // CHECK:   at {{.*}}hwasan_symbolize_stack_overflow.cpp:[[@LINE-10]]
  return 0;
}
