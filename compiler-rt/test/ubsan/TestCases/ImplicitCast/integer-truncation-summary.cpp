// RUN: %clangxx -fsanitize=implicit-integer-truncation %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NOTYPE
// RUN: %env_ubsan_opts=report_error_type=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-TYPE
// REQUIRES: !ubsan-standalone && !ubsan-standalone-static

#include <stdint.h>

int main() {
  uint8_t t0 = (~(uint32_t(0)));
  // CHECK-NOTYPE: SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior {{.*}}summary.cpp:[[@LINE-1]]:16
  // CHECK-TYPE: SUMMARY: UndefinedBehaviorSanitizer: implicit-integer-truncation {{.*}}summary.cpp:[[@LINE-2]]:16
  return 0;
}
