// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// When built as C on Linux, strndup is transformed to __strndup.
// RUN: %clangxx_asan -O3 -xc %s -o %t && not %run %t 2>&1 | FileCheck %s

// Unwind problem on arm: "main" is missing from the allocation stack trace.
// UNSUPPORTED: win32,s390,armv7l-unknown-linux-gnueabihf

#include <string.h>

char kString[] = "foo";

int main(int argc, char **argv) {
  char *copy = strndup(kString, 2);
  int x = copy[2 + argc];  // BOOM
  // CHECK: AddressSanitizer: heap-buffer-overflow
  // CHECK: #0 {{.*}}main {{.*}}strndup_oob_test.cc:[[@LINE-2]]
  // CHECK-LABEL: allocated by thread T{{.*}} here:
  // CHECK: #{{[01]}} {{.*}}strndup
  // CHECK: #{{.*}}main {{.*}}strndup_oob_test.cc:[[@LINE-6]]
  // CHECK-LABEL: SUMMARY
  // CHECK: strndup_oob_test.cc:[[@LINE-7]]
  return x;
}
