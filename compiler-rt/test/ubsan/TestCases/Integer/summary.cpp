// RUN: %clangxx -fsanitize=integer %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NOTYPE
// RUN: %env_ubsan_opts=report_error_type=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-TYPE
// REQUIRES: ubsan-asan

#include <stdint.h>

int main() {
  (void)(uint64_t(10000000000000000000ull) + uint64_t(9000000000000000000ull));
  // CHECK-NOTYPE: SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior {{.*}}summary.cpp:[[@LINE-1]]:44
  // CHECK-TYPE: SUMMARY: UndefinedBehaviorSanitizer: unsigned-integer-overflow {{.*}}summary.cpp:[[@LINE-2]]:44
  return 0;
}
