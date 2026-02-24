// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_ffsoi2
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI int __ffsoi2(oi_int a);

int test__ffsoi2(oi_int a, int expected) {
  int x = __ffsoi2(a);
  if (x != expected) {
    printf("error in __ffsoi2: expected %d, got %d\n", expected, x);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  // Zero
  if (test__ffsoi2((oi_int)0, 0))
    return 1;
  // 1 (bit 0 set)
  if (test__ffsoi2((oi_int)1, 1))
    return 1;
  // 2 (bit 1 set)
  if (test__ffsoi2((oi_int)2, 2))
    return 1;
  // Bit 63 set
  if (test__ffsoi2(make_oi(make_ti(0, 0), make_ti(0, 0x8000000000000000ULL)),
                   64))
    return 1;
  // Bit 64 set
  if (test__ffsoi2(make_oi(make_ti(0, 0), make_ti(1, 0)), 65))
    return 1;
  // Bit 127 set
  if (test__ffsoi2(make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0)),
                   128))
    return 1;
  // Bit 128 set
  if (test__ffsoi2(make_oi(make_ti(0, 1), make_ti(0, 0)), 129))
    return 1;
  // Bit 191 set
  if (test__ffsoi2(make_oi(make_ti(0, 0x8000000000000000ULL), make_ti(0, 0)),
                   192))
    return 1;
  // Bit 192 set
  if (test__ffsoi2(make_oi(make_ti(1, 0), make_ti(0, 0)), 193))
    return 1;
  // Bit 255 set (MSB)
  if (test__ffsoi2(make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0)),
                   256))
    return 1;
  // All ones
  if (test__ffsoi2((oi_int)(ou_int)-1, 1))
    return 1;
  // Multiple bits, lowest is bit 8
  if (test__ffsoi2((oi_int)0xFF00, 9))
    return 1;
  // Bits in both halves, lowest in low half
  if (test__ffsoi2(make_oi(make_ti(0, 1), make_ti(0, 0x100)), 9))
    return 1;
  // Full-width big-number tests.
  // Expected values verified by Python arbitrary-precision arithmetic.
  if (test__ffsoi2(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          17))
    return 1;
  if (test__ffsoi2(
          make_oi(make_ti(0x1111222233334444LL, 0x5555666677778888ULL),
                  make_ti(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          1))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
