// RUN: %clang_hwasan -Wl,--build-id -g %s -o %t
// RUN: %env_hwasan_opts=symbolize=0 not %run %t 2>&1 | hwasan_symbolize --symbols $(dirname %t) --index | FileCheck %s

// REQUIRES: pointer-tagging

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
