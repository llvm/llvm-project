// Checks that dfsan works with trace-cmp instrumentation, even if some hooks
// are not defined (relies on week hooks implemented in dfsan).
//
// RUN: %clang_dfsan -fsanitize-coverage=trace-pc-guard,pc-table,func,trace-cmp %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s
//
// REQUIRES: x86_64-target-arch

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <sanitizer/dfsan_interface.h>

uint32_t a4, b4;
uint64_t a8, b8;

// Define just two hooks, and leave others undefined.
void __dfsw___sanitizer_cov_trace_const_cmp4(uint8_t a, uint8_t b,
                                             dfsan_label l1, dfsan_label l2) {
  printf("const_cmp4 %d %d\n", a, b);
}
void __dfsw___sanitizer_cov_trace_cmp8(uint8_t a, uint8_t b, dfsan_label l1,
                                       dfsan_label l2) {
  printf("cmp8 %d %d\n", a, b);
}

int main(int argc, char **argv) {
  printf("MAIN\n");
  // CHECK: MAIN

  if (a4 != b4) abort();
  if (a4 == 42) abort();
  // CHECK: const_cmp4 42 0
  if (a8 != b8) abort();
  // CHECK: cmp8 0 0
  if (a8 == 66) abort();

  switch (10 / (a4 + 2)) {
    case 1: abort();
    case 2: exit(1);
    case 5:
            printf("SWITCH OK\n");
            break;
  }
  // CHECK: SWITCH OK


  printf("DONE\n");
  // CHECK: DONE
  return 0;
}
