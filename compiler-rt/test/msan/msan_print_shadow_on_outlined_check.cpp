// RUN: %clangxx_msan -fsanitize-recover=memory -mllvm -msan-instrumentation-with-call-threshold=0 -g %s -o %t \
// RUN:   && not env MSAN_OPTIONS=verbosity=1 %run %t 2>&1 | FileCheck %s

// REQUIRES: x86_64-target-arch
// 'long double' implementation varies between platforms.

#include <ctype.h>
#include <stdio.h>

#include <sanitizer/msan_interface.h>

int main(int argc, char *argv[]) {
  long double a;
  printf("a: %Lf\n", a);
  // CHECK: Shadow value (16 bytes): ffffffff ffffffff ffff0000 00000000

  unsigned long long b;
  printf("b: %llu\n", b);
  // CHECK: Shadow value (8 bytes): ffffffff ffffffff

  char *p = (char *)(&b);
  p[2] = 36;
  printf("b: %lld\n", b);
  // CHECK: Shadow value (8 bytes): ffff00ff ffffffff

  b = b << 8;
  printf("b: %lld\n", b);
  __msan_print_shadow(&b, sizeof(b));
  // CHECK: Shadow value (8 bytes): 00ffff00 ffffffff

  unsigned int c;
  printf("c: %u\n", c);
  // CHECK: Shadow value (4 bytes): ffffffff

  // Converted to boolean
  if (c) {
    // CHECK: Shadow value (1 byte): 01
    printf("Hello\n");
  }

  return 0;
}
