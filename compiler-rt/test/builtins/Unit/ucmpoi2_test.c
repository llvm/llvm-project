// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_ucmpoi2
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI si_int __ucmpoi2(ou_int a, ou_int b);

int test__ucmpoi2(ou_int a, ou_int b, si_int expected) {
  si_int x = __ucmpoi2(a, b);
  if (x != expected) {
    printf("error in __ucmpoi2: expected %d, got %d\n", expected, x);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  // Equal
  if (test__ucmpoi2((ou_int)0, (ou_int)0, 1))
    return 1;
  if (test__ucmpoi2((ou_int)1, (ou_int)1, 1))
    return 1;
  // Less than
  if (test__ucmpoi2((ou_int)0, (ou_int)1, 0))
    return 1;
  // Greater than
  if (test__ucmpoi2((ou_int)1, (ou_int)0, 2))
    return 1;
  // All-ones is maximum unsigned
  if (test__ucmpoi2((ou_int)-1, (ou_int)0, 2))
    return 1;
  if (test__ucmpoi2((ou_int)0, (ou_int)-1, 0))
    return 1;
  // High half comparison
  if (test__ucmpoi2(make_ou(make_tu(0, 1), make_tu(0, 0)),
                    make_ou(make_tu(0, 0), make_tu(-1, -1)), 2))
    return 1;
  // Large equal values
  {
    ou_int big = make_ou(make_tu(0x1234, 0x5678), make_tu(0x9ABC, 0xDEF0));
    if (test__ucmpoi2(big, big, 1))
      return 1;
  }
  // MAX > 0
  if (test__ucmpoi2((ou_int)-1, (ou_int)0, 2))
    return 1;
  // Differ only in low half
  if (test__ucmpoi2(make_ou(make_tu(0, 1), make_tu(0, 1)),
                    make_ou(make_tu(0, 1), make_tu(0, 2)), 0))
    return 1;
  if (test__ucmpoi2(make_ou(make_tu(0, 1), make_tu(0, 2)),
                    make_ou(make_tu(0, 1), make_tu(0, 1)), 2))
    return 1;
  // Differ only in highest 64-bit word
  if (test__ucmpoi2(make_ou(make_tu(1, 0), make_tu(0, 0)),
                    make_ou(make_tu(2, 0), make_tu(0, 0)), 0))
    return 1;
  if (test__ucmpoi2(make_ou(make_tu(2, 0), make_tu(0, 0)),
                    make_ou(make_tu(1, 0), make_tu(0, 0)), 2))
    return 1;
  // Adjacent values
  if (test__ucmpoi2((ou_int)100, (ou_int)101, 0))
    return 1;
  if (test__ucmpoi2((ou_int)101, (ou_int)100, 2))
    return 1;
  // Full-width big-number test (all 4 limbs populated).
  // A > B unsigned.
  // Expected value verified by Python arbitrary-precision arithmetic.
  if (test__ucmpoi2(
          make_ou(make_tu(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_tu(0x3333444455556666ULL, 0x7777888899990000ULL)),
          make_ou(make_tu(0x1111222233334444LL, 0x5555666677778888ULL),
                  make_tu(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          2))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
