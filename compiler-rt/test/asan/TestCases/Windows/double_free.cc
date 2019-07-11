// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <malloc.h>

int main() {
  int *x = (int*)malloc(42 * sizeof(int));
  free(x);
  free(x);
// CHECK: AddressSanitizer: attempting double-free on [[ADDR:0x[0-9a-f]+]]
// CHECK-NEXT: {{#0 .* free }}
// CHECK-NEXT: {{#1 .* main .*double_free.cc}}:[[@LINE-3]]
// CHECK: [[ADDR]] is located 0 bytes inside of 168-byte region
// CHECK-LABEL: freed by thread T0 here:
// CHECK-NEXT: {{#0 .* free }}
// CHECK-NEXT: {{#1 .* main .*double_free.cc}}:[[@LINE-8]]
// CHECK-LABEL: previously allocated by thread T0 here:
// CHECK-NEXT: {{#0 .* malloc }}
// CHECK-NEXT: {{#1 .* main .*double_free.cc}}:[[@LINE-12]]
  return 0;
}
