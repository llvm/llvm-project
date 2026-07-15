// RUN: %clang -x c -fsanitize=pointer-overflow %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-NOTYPE
// RUN: %env_ubsan_opts=report_error_type=1 %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-TYPE

// RUN: %clangxx -fsanitize=pointer-overflow %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-NOTYPE
// RUN: %env_ubsan_opts=report_error_type=1 %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-TYPE

#include <stdlib.h>

int main(int argc, char *argv[]) {
  char *base, *result;

  base = (char *)0;
  result = base + 0;
  // CHECK-NOTYPE-NOT: SUMMARY:
  // CHECK-TYPE-NOT: SUMMARY:

  base = (char *)0;
  result = base + 1;
  // CHECK-NOTYPE: SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior {{.*}}summary.cpp:[[@LINE-1]]:17
  // CHECK-TYPE: SUMMARY: UndefinedBehaviorSanitizer: nullptr-with-nonzero-offset {{.*}}summary.cpp:[[@LINE-2]]:17

  base = (char *)1;
  result = base - 1;
  // CHECK-NOTYPE: SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior {{.*}}summary.cpp:[[@LINE-1]]:17
  // CHECK-TYPE: SUMMARY: UndefinedBehaviorSanitizer: nullptr-after-nonzero-offset {{.*}}summary.cpp:[[@LINE-2]]:17

  return 0;
}
