// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_parityoi2
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI int __parityoi2(oi_int a);

int test__parityoi2(oi_int a, int expected) {
  int x = __parityoi2(a);
  if (x != expected) {
    printf("error in __parityoi2: expected %d, got %d\n", expected, x);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  // Zero (even parity)
  if (test__parityoi2((oi_int)0, 0))
    return 1;
  // One (odd parity)
  if (test__parityoi2((oi_int)1, 1))
    return 1;
  // Two bits set (even parity)
  if (test__parityoi2((oi_int)3, 0))
    return 1;
  // Three bits set (odd parity)
  if (test__parityoi2((oi_int)7, 1))
    return 1;
  // All ones = 256 bits set (even parity)
  if (test__parityoi2((oi_int)(ou_int)-1, 0))
    return 1;
  // One bit in high half (odd parity)
  if (test__parityoi2(make_oi(make_ti(0, 1), make_ti(0, 0)), 1))
    return 1;
  // One bit in each half (even parity)
  if (test__parityoi2(make_oi(make_ti(0, 1), make_ti(0, 1)), 0))
    return 1;
  // High half all ones (128 bits = even), low half zero
  if (test__parityoi2(make_oi(make_ti(-1, -1), make_ti(0, 0)), 0))
    return 1;
  // 0xFF (8 bits = even parity)
  if (test__parityoi2((oi_int)0xFF, 0))
    return 1;
  // 0x7F (7 bits = odd parity)
  if (test__parityoi2((oi_int)0x7F, 1))
    return 1;
  // MSB only (odd parity)
  if (test__parityoi2(make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0)),
                      1))
    return 1;
  // One bit in each 64-bit word (4 bits = even parity)
  if (test__parityoi2(make_oi(make_ti(1, 1), make_ti(1, 1)), 0))
    return 1;
  // Three bits across multiple words (odd parity)
  if (test__parityoi2(make_oi(make_ti(1, 1), make_ti(1, 0)), 1))
    return 1;
  // Full-width big-number tests.
  // Expected values verified by Python arbitrary-precision arithmetic.
  if (test__parityoi2(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          0))
    return 1;
  if (test__parityoi2(
          make_oi(make_ti(0x1111222233334444LL, 0x5555666677778888ULL),
                  make_ti(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          0))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
