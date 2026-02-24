// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_ctzoi2
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI int __ctzoi2(oi_int a);

int test__ctzoi2(oi_int a, int expected) {
  int x = __ctzoi2(a);
  if (x != expected) {
    printf("error in __ctzoi2: expected %d, got %d\n", expected, x);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  // 1
  if (test__ctzoi2((oi_int)1, 0))
    return 1;
  // 2
  if (test__ctzoi2((oi_int)2, 1))
    return 1;
  // Bit at position 63
  if (test__ctzoi2(make_oi(make_ti(0, 0), make_ti(0, 0x8000000000000000ULL)),
                   63))
    return 1;
  // Bit at position 64
  if (test__ctzoi2(make_oi(make_ti(0, 0), make_ti(1, 0)), 64))
    return 1;
  // Bit at position 127
  if (test__ctzoi2(make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0)),
                   127))
    return 1;
  // Bit at position 128
  if (test__ctzoi2(make_oi(make_ti(0, 1), make_ti(0, 0)), 128))
    return 1;
  // Bit at position 191
  if (test__ctzoi2(make_oi(make_ti(0, 0x8000000000000000ULL), make_ti(0, 0)),
                   191))
    return 1;
  // Bit at position 192
  if (test__ctzoi2(make_oi(make_ti(1, 0), make_ti(0, 0)), 192))
    return 1;
  // Bit at position 255 (MSB)
  if (test__ctzoi2(make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0)),
                   255))
    return 1;
  // All ones
  if (test__ctzoi2((oi_int)(ou_int)-1, 0))
    return 1;
  // Multiple bits, lowest is position 8
  if (test__ctzoi2((oi_int)0xFF00, 8))
    return 1;
  // Bits in both halves, lowest in low half
  if (test__ctzoi2(make_oi(make_ti(0, 1), make_ti(0, 0x100)), 8))
    return 1;
  // Full-width big-number tests.
  // Expected values verified by Python arbitrary-precision arithmetic.
  if (test__ctzoi2(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          16))
    return 1;
  if (test__ctzoi2(
          make_oi(make_ti(0x1111222233334444LL, 0x5555666677778888ULL),
                  make_ti(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          0))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
