// RUN: %clangxx -fsanitize=signed-integer-overflow %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdint.h>

int main() {
  // These promote to 'int'.
  (void)(int8_t(-2) * int8_t(0x7f));
  (void)(int16_t(0x7fff) * int16_t(0x7fff));
  (void)(uint16_t(0xffff) * int16_t(0x7fff));
  (void)(uint16_t(0xffff) * uint16_t(0x8000));

  // CHECK: mul-overflow.cpp:13:27: runtime error: signed integer overflow: 65535 * 32769 cannot be represented in type 'int'
  (void)(uint16_t(0xffff) * uint16_t(0x8001));
}
