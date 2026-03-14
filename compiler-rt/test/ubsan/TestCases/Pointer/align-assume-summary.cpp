// RUN: %clangxx -fsanitize=alignment %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NOTYPE
// RUN: %env_ubsan_opts=report_error_type=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-TYPE

#include <stdlib.h>

int main(int argc, char* argv[]) {
  char *ptr = (char *)malloc(2);

  __builtin_assume_aligned(ptr + 1, 0x8000);
  // CHECK-NOTYPE: SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior {{.*}}summary.cpp:[[@LINE-1]]:32
  // CHECK-TYPE: SUMMARY: UndefinedBehaviorSanitizer: alignment-assumption {{.*}}summary.cpp:[[@LINE-2]]:32
  free(ptr);

  return 0;
}
