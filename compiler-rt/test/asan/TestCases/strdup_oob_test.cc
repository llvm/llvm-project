// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// When built as C on Linux, strdup is transformed to __strdup.
// RUN: %clangxx_asan -O3 -xc %s -o %t && not %run %t 2>&1 | FileCheck %s

// Unwind problem on arm: "main" is missing from the allocation stack trace.
// REQUIRES: (arm-target-arch || armhf-target-arch), fast-unwinder-works

// FIXME: We fail to intercept strdup with the dynamic WinASan RTL, so it's not
// in the stack trace.
// XFAIL: win32-dynamic-asan

#include <string.h>

char kString[] = "foo";

int main(int argc, char **argv) {
  char *copy = strdup(kString);
  int x = copy[4 + argc];  // BOOM
  // CHECK: AddressSanitizer: heap-buffer-overflow
  // CHECK: #0 {{.*}}main {{.*}}strdup_oob_test.cc:[[@LINE-2]]
  // CHECK-LABEL: allocated by thread T{{.*}} here:
  // CHECK: #{{[01]}} {{.*}}strdup
  // CHECK: #{{.*}}main {{.*}}strdup_oob_test.cc:[[@LINE-6]]
  // CHECK-LABEL: SUMMARY
  // CHECK: strdup_oob_test.cc:[[@LINE-7]]
  return x;
}
