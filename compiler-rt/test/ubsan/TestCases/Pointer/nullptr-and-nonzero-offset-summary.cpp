// RUN: %clang -x c -fsanitize=pointer-overflow %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-NOTYPE,CHECK-NOTYPE-C
// RUN: %env_ubsan_opts=report_error_type=1 %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-TYPE,CHECK-TYPE-C

// RUN: %clangxx -fsanitize=pointer-overflow %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-NOTYPE,CHECK-NOTYPE-CPP
// RUN: %env_ubsan_opts=report_error_type=1 %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-TYPE,CHECK-TYPE-CPP

// REQUIRES: !ubsan-standalone && !ubsan-standalone-static

#include <stdlib.h>

int main(int argc, char *argv[]) {
  char *base, *result;

  base = (char *)0;
  result = base + 0;
  // CHECK-NOTYPE-C: SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior {{.*}}summary.cpp:[[@LINE-1]]:17
  // CHECK-TYPE-C: SUMMARY: UndefinedBehaviorSanitizer: nullptr-with-offset {{.*}}summary.cpp:[[@LINE-2]]:17
  // CHECK-NOTYPE-CPP-NOT: SUMMARY:
  // CHECK-TYPE-CPP-NOT: SUMMARY:

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
