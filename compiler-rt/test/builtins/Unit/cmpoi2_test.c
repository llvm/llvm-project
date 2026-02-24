// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_cmpoi2
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI si_int __cmpoi2(oi_int a, oi_int b);

int test__cmpoi2(oi_int a, oi_int b, si_int expected) {
  si_int x = __cmpoi2(a, b);
  if (x != expected) {
    printf("error in __cmpoi2: expected %d, got %d\n", expected, x);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  // Equal
  if (test__cmpoi2((oi_int)0, (oi_int)0, 1))
    return 1;
  if (test__cmpoi2((oi_int)1, (oi_int)1, 1))
    return 1;
  if (test__cmpoi2((oi_int)-1, (oi_int)-1, 1))
    return 1;
  // Less than
  if (test__cmpoi2((oi_int)0, (oi_int)1, 0))
    return 1;
  if (test__cmpoi2((oi_int)-1, (oi_int)0, 0))
    return 1;
  // Greater than
  if (test__cmpoi2((oi_int)1, (oi_int)0, 2))
    return 1;
  if (test__cmpoi2((oi_int)0, (oi_int)-1, 2))
    return 1;
  // Large values: high half > low half
  if (test__cmpoi2(make_oi(make_ti(0, 1), make_ti(0, 0)),
                   make_oi(make_ti(0, 0), make_ti(-1, -1)), 2))
    return 1;
  // Large equal values
  {
    oi_int big = make_oi(make_ti(0x1234, 0x5678), make_ti(0x9ABC, 0xDEF0));
    if (test__cmpoi2(big, big, 1))
      return 1;
  }
  // MAX > 0
  if (test__cmpoi2(make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1)),
                   (oi_int)0, 2))
    return 1;
  // MIN < 0
  if (test__cmpoi2(make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0)),
                   (oi_int)0, 0))
    return 1;
  // MIN < MAX
  if (test__cmpoi2(make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0)),
                   make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1)),
                   0))
    return 1;
  // Differ only in low half
  if (test__cmpoi2(make_oi(make_ti(0, 1), make_ti(0, 1)),
                   make_oi(make_ti(0, 1), make_ti(0, 2)), 0))
    return 1;
  if (test__cmpoi2(make_oi(make_ti(0, 1), make_ti(0, 2)),
                   make_oi(make_ti(0, 1), make_ti(0, 1)), 2))
    return 1;
  // Negative values: -1 > -2
  if (test__cmpoi2((oi_int)-1, (oi_int)-2, 2))
    return 1;
  if (test__cmpoi2((oi_int)-2, (oi_int)-1, 0))
    return 1;
  // Full-width big-number test (all 4 limbs populated).
  // A is negative signed, B is positive signed, so A < B.
  // Expected value verified by Python arbitrary-precision arithmetic.
  if (test__cmpoi2(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          make_oi(make_ti(0x1111222233334444LL, 0x5555666677778888ULL),
                  make_ti(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          0))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
