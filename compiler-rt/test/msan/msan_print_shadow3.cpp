// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdint.h>
#include <sanitizer/msan_interface.h>

int main(void) {
  unsigned long long x = 0; // For 8-byte alignment.
  char x_s[4] = {0x77, 0x65, 0x43, 0x21};
  __msan_partial_poison(&x, &x_s, sizeof(x_s));
  __msan_print_shadow(&x, sizeof(x_s));
  return 0;
}

// CHECK: Shadow map [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}) of [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}), 4 bytes:
// CHECK: 0x{{.*}}: 77654321 ........ ........ ........
