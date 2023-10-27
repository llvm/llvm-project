// RUN: %clang -w -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: hwasan
// UNSUPPORTED: lsan
// UNSUPPORTED: msan
// UNSUPPORTED: ubsan
#include <stdio.h>
int main() {
  int a;
  printf("%Q\n", 1);
  printf("%Q\n", 1);
  printf("%Q\n", 1);
}
// CHECK: unexpected format specifier in printf interceptor: %Q (reported once per process)
// CHECK-NOT: unexpected format specifier in printf interceptor
