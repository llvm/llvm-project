// RUN: %clang -fPIC -shared -O2 -D_FORTIFY_SOURCE=2 -D_DSO %s -o %t.so
// RUN: %clang_asan %s -o %t %t.so
// RUN: not %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: android
#ifdef _DSO
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
__attribute__((noinline)) int foo() {
  char *write_buffer = (char *)malloc(1);
  // CHECK: AddressSanitizer: heap-buffer-overflow
  snprintf(write_buffer, 4096, "%s_%s", "one", "two");
  return write_buffer[0];
}
#else
extern int foo();
int main() { return foo(); }
#endif
