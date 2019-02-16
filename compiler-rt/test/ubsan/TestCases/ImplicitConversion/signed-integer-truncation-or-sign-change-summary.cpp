// RUN: %clangxx -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NOTYPE
// RUN: %env_ubsan_opts=report_error_type=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-TYPE
// REQUIRES: !ubsan-standalone && !ubsan-standalone-static

#include <stdint.h>

int main() {
  int8_t t0 = (~(uint32_t(0)));
  // CHECK-NOTYPE: SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior {{.*}}summary.cpp:[[@LINE-1]]:15
  // CHECK-TYPE: SUMMARY: UndefinedBehaviorSanitizer: implicit-signed-integer-truncation-or-sign-change {{.*}}summary.cpp:[[@LINE-2]]:15
  return 0;
}
