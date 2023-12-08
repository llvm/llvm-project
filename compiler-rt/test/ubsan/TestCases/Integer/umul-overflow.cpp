// RUN: %clangxx -fsanitize=unsigned-integer-overflow %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdint.h>

int main() {
  // These promote to 'int'.
  (void)(int8_t(-2) * int8_t(0x7f));
  (void)(int16_t(0x7fff) * int16_t(0x7fff));
  (void)(uint16_t(0xffff) * int16_t(0x7fff));
  (void)(uint16_t(0xffff) * uint16_t(0x8000));

  // Not an unsigned overflow
  (void)(uint16_t(0xffff) * uint16_t(0x8001));

  (void)(uint32_t(0xffffffff) * uint32_t(0x2));
  // CHECK: umul-overflow.cpp:15:31: runtime error: unsigned integer overflow: 4294967295 * 2 cannot be represented in type '{{uint32_t|unsigned int}}'

  return 0;
}
