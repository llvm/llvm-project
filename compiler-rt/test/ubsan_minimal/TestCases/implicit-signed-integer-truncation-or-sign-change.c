// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK

#include <stdint.h>

int main() {
// CHECK-NOT: implicit-conversion

  // Explicitly casting hides it,
  int8_t n0 = (int8_t)((uint32_t)-1);

  // Positive tests.
  int8_t t0 = (uint32_t)-1;
// CHECK: implicit-conversion by 0x{{[[:xdigit:]]+$}}
// CHECK-NOT: implicit-conversion

  return 0;
}
