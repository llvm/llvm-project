// UNSUPPORTED: target={{.*-windows-gnu}}

// Make sure symbolization works even if the path to the .exe file changes.
// RUN: mkdir %t || true
// RUN: %clang_cl_asan -Od %s -Fe%t/symbols_path.exe
// RUN: not %run %t/symbols_path.exe 2>&1 | FileCheck %s
// RUN: mkdir %t2 || true
// RUN: mv %t/* %t2
// RUN: not %run %t2/symbols_path.exe 2>&1 | FileCheck %s

#include <malloc.h>

int main() {
  char *buffer = (char*)malloc(42);
  buffer[-1] = 42;
// CHECK: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] thread T0
// CHECK-NEXT: {{#0 .* main .*symbols_path.cpp}}:[[@LINE-3]]
// CHECK: [[ADDR]] is located 1 bytes before 42-byte region
// CHECK: allocated by thread T0 here:
// CHECK-NEXT: {{#0 .* malloc}}
// CHECK-NEXT: {{#1 .* main .*symbols_path.cpp}}:[[@LINE-8]]
  free(buffer);
}
