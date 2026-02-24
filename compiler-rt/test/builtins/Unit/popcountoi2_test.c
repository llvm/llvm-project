// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_popcountoi2
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI int __popcountoi2(oi_int a);

int test__popcountoi2(oi_int a, int expected) {
  int x = __popcountoi2(a);
  if (x != expected) {
    printf("error in __popcountoi2: expected %d, got %d\n", expected, x);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  // Zero
  if (test__popcountoi2((oi_int)0, 0))
    return 1;
  // One
  if (test__popcountoi2((oi_int)1, 1))
    return 1;
  // All ones (256 bits)
  if (test__popcountoi2((oi_int)(ou_int)-1, 256))
    return 1;
  // 0xFF (8 bits set)
  if (test__popcountoi2((oi_int)0xFF, 8))
    return 1;
  // One bit in each 128-bit half
  if (test__popcountoi2(make_oi(make_ti(0, 1), make_ti(0, 1)), 2))
    return 1;
  // 0xFF in high half only
  if (test__popcountoi2(make_oi(make_ti(0, 0xFF), make_ti(0, 0)), 8))
    return 1;
  // Alternating bits (0xAA...AA) = 128 bits set
  if (test__popcountoi2(
          make_oi(make_ti(0xAAAAAAAAAAAAAAAALL, 0xAAAAAAAAAAAAAAAALL),
                  make_ti(0xAAAAAAAAAAAAAAAALL, 0xAAAAAAAAAAAAAAAALL)),
          128))
    return 1;
  // Alternating bits (0x55...55) = 128 bits set
  if (test__popcountoi2(
          make_oi(make_ti(0x5555555555555555LL, 0x5555555555555555LL),
                  make_ti(0x5555555555555555LL, 0x5555555555555555LL)),
          128))
    return 1;
  // One bit in each 64-bit word (4 bits total)
  if (test__popcountoi2(make_oi(make_ti(1, 1), make_ti(1, 1)), 4))
    return 1;
  // High half all ones, low half zero = 128
  if (test__popcountoi2(make_oi(make_ti(-1, -1), make_ti(0, 0)), 128))
    return 1;
  // Low half all ones, high half zero = 128
  if (test__popcountoi2(make_oi(make_ti(0, 0), make_ti(-1, -1)), 128))
    return 1;
  // Single high bit = 1
  if (test__popcountoi2(
          make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0)), 1))
    return 1;
  // Full-width big-number tests.
  // Expected values verified by Python arbitrary-precision arithmetic.
  if (test__popcountoi2(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          128))
    return 1;
  if (test__popcountoi2(
          make_oi(make_ti(0x1111222233334444LL, 0x5555666677778888ULL),
                  make_ti(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          132))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
