// RUN: rm -rf %t; mkdir %t
// RUN: %clang_hwasan -Wl,--build-id -g %s -o %t/hwasan_uas
// RUN: %env_hwasan_opts=symbolize=0 not %run %t/hwasan_uas 2>&1 | hwasan_symbolize --symbols %t --index | FileCheck %s

// This doesn't work on X86, because that uses instrument-with-calls which
// disables frame records.
// REQUIRES: aarch64-target-arch

#include <sanitizer/hwasan_interface.h>
#include <stdlib.h>

static volatile char sink;

int main(int argc, char **argv) {
  volatile char *y;
  {
    volatile char x[10];
    y = &x[0];
  }
  sink = *y;
  // CHECK: Potentially referenced stack object:
  // CHECK:   0 bytes inside a variable "x" in stack frame of function "main"
  // CHECK:   at {{.*}}hwasan_symbolize_stack_uas.cpp:[[@LINE-6]]
  return 0;
}
