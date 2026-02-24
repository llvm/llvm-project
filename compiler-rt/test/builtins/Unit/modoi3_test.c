// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_modoi3
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __modoi3(oi_int a, oi_int b);

int test__modoi3(oi_int a, oi_int b, oi_int expected) {
  oi_int x = __modoi3(a, b);
  if (x != expected) {
    printf("error in __modoi3\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__modoi3((oi_int)0, (oi_int)1, (oi_int)0))
    return 1;
  if (test__modoi3((oi_int)10, (oi_int)3, (oi_int)1))
    return 1;
  if (test__modoi3((oi_int)-10, (oi_int)3, (oi_int)-1))
    return 1;
  if (test__modoi3((oi_int)10, (oi_int)-3, (oi_int)1))
    return 1;
  if (test__modoi3((oi_int)-10, (oi_int)-3, (oi_int)-1))
    return 1;
  if (test__modoi3((oi_int)100, (oi_int)7, (oi_int)2))
    return 1;
  // Exact division has zero remainder
  if (test__modoi3((oi_int)42, (oi_int)42, (oi_int)0))
    return 1;
  // Dividend smaller than divisor
  if (test__modoi3((oi_int)3, (oi_int)10, (oi_int)3))
    return 1;
  if (test__modoi3((oi_int)-3, (oi_int)10, (oi_int)-3))
    return 1;
  // Large value in high half mod small
  // (1 << 128) % 3 = 1 (since 2^128 mod 3 = 1)
  if (test__modoi3(make_oi(make_ti(0, 1), make_ti(0, 0)), (oi_int)3, (oi_int)1))
    return 1;
  // (1 << 128) % 2 = 0
  if (test__modoi3(make_oi(make_ti(0, 1), make_ti(0, 0)), (oi_int)2, (oi_int)0))
    return 1;
  // Negative large value mod
  if (test__modoi3(make_oi(make_ti(-1, -1), make_ti(0, 0)), (oi_int)3,
                   (oi_int)-1))
    return 1;
  // Cross-half boundary value mod small
  if (test__modoi3(make_oi(make_ti(0, 1), make_ti(0, 5)), (oi_int)4, (oi_int)1))
    return 1;
  // Large mod large (same value)
  {
    oi_int big = make_oi(make_ti(0, 0x100), make_ti(0, 0));
    if (test__modoi3(big, big, (oi_int)0))
      return 1;
  }
  // Full-width big-number test (all 4 limbs populated).
  // A(signed) % B(signed), verified by Python: q*b + r == a.
  // Expected value verified by Python arbitrary-precision arithmetic.
  if (test__modoi3(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          make_oi(make_ti(0x1111222233334444LL, 0x5555666677778888ULL),
                  make_ti(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          make_oi(make_ti(0xEEEF44449999EEEFLL, 0x44449998EEEF4444ULL),
                  make_ti(0x9999EEEF44449999ULL, 0xEEEF444499954444ULL))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
