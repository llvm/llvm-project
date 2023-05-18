// RUN: %clangxx -fsanitize=signed-integer-overflow %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdint.h>

int main() {
  unsigned(0x80000000) / -1;

  // CHECK: div-overflow.cpp:9:23: runtime error: division of -2147483648 by -1 cannot be represented in type 'int'
  int32_t(0x80000000) / -1;
}
