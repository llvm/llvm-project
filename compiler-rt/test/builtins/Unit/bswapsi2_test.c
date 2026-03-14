// UNSUPPORTED: armv6m-target-arch
// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_bswapsi2

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern uint32_t __bswapsi2(uint32_t);

int test__bswapsi2(uint32_t a, uint32_t expected) {
  uint32_t actual = __bswapsi2(a);
  if (actual != expected)
    printf("error in test__bswapsi2(0x%0X) = 0x%0X, expected 0x%0X\n", a,
           actual, expected);
  return actual != expected;
}

int main() {
  if (test__bswapsi2(0x12345678, 0x78563412))
    return 1;
  if (test__bswapsi2(0x00000001, 0x01000000))
    return 1;
  return 0;
}
