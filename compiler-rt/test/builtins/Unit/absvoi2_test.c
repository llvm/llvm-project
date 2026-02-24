// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_absvoi2
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __absvoi2(oi_int a);

int test__absvoi2(oi_int a, oi_int expected) {
  oi_int x = __absvoi2(a);
  if (x != expected) {
    printf("error in __absvoi2\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__absvoi2((oi_int)0, (oi_int)0))
    return 1;
  if (test__absvoi2((oi_int)1, (oi_int)1))
    return 1;
  if (test__absvoi2((oi_int)-1, (oi_int)1))
    return 1;
  if (test__absvoi2((oi_int)42, (oi_int)42))
    return 1;
  if (test__absvoi2((oi_int)-42, (oi_int)42))
    return 1;
  // Large positive value (already positive, no change)
  {
    oi_int big = make_oi(make_ti(0, 1), make_ti(0, 0));
    if (test__absvoi2(big, big))
      return 1;
  }
  // Large negative value
  if (test__absvoi2(make_oi(make_ti(-1, -1), make_ti(0, 0)),
                    make_oi(make_ti(0, 1), make_ti(0, 0))))
    return 1;
  // MAX (already positive)
  {
    oi_int MAX = make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1));
    if (test__absvoi2(MAX, MAX))
      return 1;
  }
  // Note: MIN would abort, so we don't test it.
  // Full-width big-number test (all 4 limbs populated).
  // Expected value verified by Python arbitrary-precision arithmetic.
  // C is negative signed; abs(C) = |C|
  if (test__absvoi2(
          make_oi(make_ti(0xDDDDEEEEFFFF0000LL, 0x1111222233334444ULL),
                  make_ti(0x5555666677778888ULL, 0x9999AAAABBBBCCCCULL)),
          make_oi(make_ti(0x222211110000FFFFLL, 0xEEEEDDDDCCCCBBBBULL),
                  make_ti(0xAAAA999988887777ULL, 0x6666555544443334ULL))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
