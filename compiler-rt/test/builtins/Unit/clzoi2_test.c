// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_clzoi2
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI int __clzoi2(oi_int a);

int test__clzoi2(oi_int a, int expected) {
  int x = __clzoi2(a);
  if (x != expected) {
    printf("error in __clzoi2: expected %d, got %d\n", expected, x);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  // Single bit in MSB position
  if (test__clzoi2(make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0)), 0))
    return 1;
  // Single bit in high half, lower position (bit 128)
  if (test__clzoi2(make_oi(make_ti(0, 1), make_ti(0, 0)), 127))
    return 1;
  // Single bit at position 128 (MSB of low half)
  if (test__clzoi2(make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0)),
                   128))
    return 1;
  // 1
  if (test__clzoi2((oi_int)1, 255))
    return 1;
  // All ones
  if (test__clzoi2((oi_int)(ou_int)-1, 0))
    return 1;
  // Value in high word only
  if (test__clzoi2(make_oi(make_ti(0, 0xFFLL), make_ti(0, 0)), 120))
    return 1;
  // Bit at position 64 (second 64-bit word)
  if (test__clzoi2(make_oi(make_ti(0, 0), make_ti(1, 0)), 191))
    return 1;
  // Bit at position 192
  if (test__clzoi2(make_oi(make_ti(1, 0), make_ti(0, 0)), 63))
    return 1;
  // 0xFF in low word only
  if (test__clzoi2((oi_int)0xFF, 248))
    return 1;
  // Single bit at position 191
  if (test__clzoi2(make_oi(make_ti(0, 0x8000000000000000ULL), make_ti(0, 0)),
                   64))
    return 1;
  // Power of 2 at position 200
  if (test__clzoi2(make_oi(make_ti(0x100LL, 0), make_ti(0, 0)), 55))
    return 1;
  // Full-width big-number tests.
  // Expected values verified by Python arbitrary-precision arithmetic.
  if (test__clzoi2(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          0))
    return 1;
  if (test__clzoi2(
          make_oi(make_ti(0x1111222233334444LL, 0x5555666677778888ULL),
                  make_ti(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          3))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
