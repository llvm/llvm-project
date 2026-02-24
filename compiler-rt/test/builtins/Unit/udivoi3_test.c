// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_udivoi3
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI ou_int __udivoi3(ou_int a, ou_int b);

int test__udivoi3(ou_int a, ou_int b, ou_int expected) {
  ou_int x = __udivoi3(a, b);
  if (x != expected) {
    printf("error in __udivoi3\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__udivoi3((ou_int)0, (ou_int)1, (ou_int)0))
    return 1;
  if (test__udivoi3((ou_int)1, (ou_int)1, (ou_int)1))
    return 1;
  if (test__udivoi3((ou_int)10, (ou_int)3, (ou_int)3))
    return 1;
  if (test__udivoi3((ou_int)100, (ou_int)7, (ou_int)14))
    return 1;
  if (test__udivoi3((ou_int)42, (ou_int)42, (ou_int)1))
    return 1;
  // Dividend < divisor
  if (test__udivoi3((ou_int)3, (ou_int)10, (ou_int)0))
    return 1;
  // Large value in high half / small
  // (1 << 128) / 2 = (1 << 127)
  if (test__udivoi3(make_ou(make_tu(0, 1), make_tu(0, 0)), (ou_int)2,
                    make_ou(make_tu(0, 0), make_tu(0x8000000000000000ULL, 0))))
    return 1;
  // (1 << 128) / 3
  if (test__udivoi3(make_ou(make_tu(0, 1), make_tu(0, 0)), (ou_int)3,
                    make_ou(make_tu(0, 0), make_tu(0x5555555555555555ULL,
                                                   0x5555555555555555ULL))))
    return 1;
  // Large / large (same value)
  {
    ou_int big = make_ou(make_tu(0, 0x100), make_tu(0, 0));
    if (test__udivoi3(big, big, (ou_int)1))
      return 1;
  }
  // Large / large (double)
  {
    ou_int big = make_ou(make_tu(0, 0x100), make_tu(0, 0));
    ou_int dbl = make_ou(make_tu(0, 0x200), make_tu(0, 0));
    if (test__udivoi3(dbl, big, (ou_int)2))
      return 1;
  }
  // All-ones / 2
  if (test__udivoi3(
          (ou_int)-1, (ou_int)2,
          make_ou(make_tu(0x7FFFFFFFFFFFFFFFULL, -1), make_tu(-1, -1))))
    return 1;
  // Cross-half boundary value / small
  if (test__udivoi3(make_ou(make_tu(0, 1), make_tu(0, 4)), (ou_int)4,
                    make_ou(make_tu(0, 0), make_tu(0x4000000000000000ULL, 1))))
    return 1;
  // Very large divisor in high half
  {
    ou_int big = make_ou(make_tu(1, 0), make_tu(0, 0));
    if (test__udivoi3(big, big, (ou_int)1))
      return 1;
  }
  // Full-width big-number test (all 4 limbs populated).
  // A / B (unsigned) = 9.
  // Expected value verified by Python arbitrary-precision arithmetic.
  if (test__udivoi3(
          make_ou(make_tu(0xAAAABBBBCCCCDDDDULL, 0xEEEEFFFF11112222ULL),
                  make_tu(0x3333444455556666ULL, 0x7777888899990000ULL)),
          make_ou(make_tu(0x1111222233334444ULL, 0x5555666677778888ULL),
                  make_tu(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          (ou_int)9))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
