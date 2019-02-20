// RUN: %clang -fsanitize=implicit-integer-sign-change %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK

#include <stdint.h>

int main() {
// CHECK-NOT: implicit-conversion

  // Explicitly casting hides it,
  int32_t n0 = (int32_t)(~((uint32_t)0));

  // Positive tests.
  int32_t t0 = (~((uint32_t)0));
// CHECK: implicit-conversion
// CHECK-NOT: implicit-conversion

  return 0;
}
