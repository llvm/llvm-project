// Test for __lsan_do_leak_check(). We test it by making the leak check run
// before global destructors, which also tests compatibility with HeapChecker's
// "normal" mode (LSan runs in "strict" mode by default).
// RUN: LSAN_BASE="detect_leaks=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE not %run %t 2>&1 | FileCheck --check-prefix=CHECK-strict %s
// RUN: LSAN_OPTIONS=$LSAN_BASE not %run %t foo 2>&1 | FileCheck --check-prefix=CHECK-normal %s

#include <stdio.h>
#include <stdlib.h>
#include <sanitizer/lsan_interface.h>

struct LeakyGlobal {
  LeakyGlobal() {
    p = malloc(1337);
  }
  ~LeakyGlobal() {
    p = 0;
  }
  void *p;
};

LeakyGlobal leaky_global;

int main(int argc, char *argv[]) {
  // Register leak check to run before global destructors.
  if (argc > 1)
    atexit(&__lsan_do_leak_check);
  void *p = malloc(666);
  printf("Test alloc: %p\n", p);
  printf("Test alloc in leaky global: %p\n", leaky_global.p);
  return 0;
}

// CHECK-strict: SUMMARY: {{(Leak|Address)}}Sanitizer: 2003 byte(s) leaked in 2 allocation(s)
// CHECK-normal: SUMMARY: {{(Leak|Address)}}Sanitizer: 666 byte(s) leaked in 1 allocation(s)
