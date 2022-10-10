// RUN: %clangxx -fsanitize=unsigned-integer-overflow %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx -fsanitize=unsigned-integer-overflow -fno-sanitize-recover=all %s -o %t && not --crash %run %t 2>&1 | FileCheck %s

#include <stdint.h>

int main() {
  uint32_t k = 0x87654321;
  k += 0xedcba987;
  // CHECK: add-overflow by 0x{{[[:xdigit:]]+$}}
}
