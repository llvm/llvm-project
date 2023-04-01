// UNSUPPORTED: target={{.*-windows-gnu}}

// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <windows.h>

int main() {
  char *buffer = new char;
  delete buffer;
  *buffer = 42;
// CHECK: AddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] thread T0
// CHECK:   {{#0 .* main .*operator_new_uaf.cpp}}:[[@LINE-3]]
// CHECK: [[ADDR]] is located 0 bytes inside of 1-byte region
// CHECK-LABEL: freed by thread T0 here:
// CHECK:   {{#0 .* operator delete}}
// CHECK:   {{#1 .* main .*operator_new_uaf.cpp}}:[[@LINE-8]]
// CHECK-LABEL: previously allocated by thread T0 here:
// CHECK:   {{#0 .* operator new}}
// CHECK:   {{#1 .* main .*operator_new_uaf.cpp}}:[[@LINE-12]]
  return 0;
}

