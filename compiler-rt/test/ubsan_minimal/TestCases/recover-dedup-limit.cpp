// RUN: %clangxx -fsanitize=signed-integer-overflow -fsanitize-recover=all %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdint.h>

#define OVERFLOW  \
  x = 0x7FFFFFFE; \
  x += __LINE__

int main() {
  int32_t x;
  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow

  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow

  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow

  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow
  OVERFLOW;  // CHECK: add-overflow

  // CHECK-NOT: add-overflow
  OVERFLOW;  // CHECK: too many errors
  // CHECK-NOT: add-overflow
  OVERFLOW;
  OVERFLOW;
  OVERFLOW;
}
