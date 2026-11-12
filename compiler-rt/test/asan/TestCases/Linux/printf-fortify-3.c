// RUN: %clang -shared -fPIC -D_DSO -O2 -D_FORTIFY_SOURCE=2 %s -o %t.so
// RUN: %clang_asan %s -o %t %t.so
// RUN: not %run %t 2>&1 | FileCheck %s
// REQUIRES: glibc-2.27
#ifdef _DSO
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
__attribute__((noinline)) char foo(const char *format, ...) {
  char *write_buffer = (char *)malloc(1);
  va_list ap;
  va_start(ap, format);
  // CHECK: AddressSanitizer: heap-buffer-overflow
  vsprintf(write_buffer, format, ap);
  va_end(ap);
  return write_buffer[0];
}
#else
extern int foo(const char *format, ...);
int main() { return foo("%s_%s", "one", "two"); }
#endif
