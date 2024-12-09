// RUN: %clang -fsanitize=implicit-integer-sign-change %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK

#include <stdio.h>
#include <stdint.h>

void __ubsan_handle_implicit_conversion_minimal() {
  printf("CUSTOM_CALLBACK\n");
}

int main() {
  int32_t t0 = (~((uint32_t)0));
// CHECK: CUSTOM_CALLBACK
  return 0;
}
