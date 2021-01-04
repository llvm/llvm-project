// RUN: %clang -fPIC -shared -O2 -D_FORTIFY_SOURCE=2 -D_DSO %s -o %t.so
// RUN: %clang_asan -o %t %t.so %s
// RUN: not %run %t 2>&1 | FileCheck %s
// REQUIRES: glibc-2.27
#ifdef _DSO
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
__attribute__((noinline)) int foo() {
  char *read_buffer = (char *)malloc(1);
  // CHECK: AddressSanitizer: heap-buffer-overflow
  fprintf(stderr, read_buffer, 4096);
  return read_buffer[0];
}
#else
extern int foo();
int main() { return foo(); }
#endif
