// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_addvoi3
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __addvoi3(oi_int a, oi_int b);

int test__addvoi3(oi_int a, oi_int b, oi_int expected) {
  oi_int x = __addvoi3(a, b);
  if (x != expected) {
    printf("error in __addvoi3\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__addvoi3((oi_int)0, (oi_int)0, (oi_int)0))
    return 1;
  if (test__addvoi3((oi_int)1, (oi_int)1, (oi_int)2))
    return 1;
  if (test__addvoi3((oi_int)-1, (oi_int)1, (oi_int)0))
    return 1;
  if (test__addvoi3((oi_int)100, (oi_int)200, (oi_int)300))
    return 1;
  // Large values in low half (carry across 64-bit boundary)
  if (test__addvoi3(make_oi(make_ti(0, 0), make_ti(0, 0xFFFFFFFFFFFFFFFFULL)),
                    make_oi(make_ti(0, 0), make_ti(0, 1)),
                    make_oi(make_ti(0, 0), make_ti(1, 0))))
    return 1;
  // Carry across 128-bit boundary (low half to high half)
  if (test__addvoi3(make_oi(make_ti(0, 0), make_ti(-1, -1)),
                    make_oi(make_ti(0, 0), make_ti(0, 1)),
                    make_oi(make_ti(0, 1), make_ti(0, 0))))
    return 1;
  // Negative + negative
  if (test__addvoi3((oi_int)-100, (oi_int)-200, (oi_int)-300))
    return 1;
  // Large positive values
  if (test__addvoi3(make_oi(make_ti(0, 1), make_ti(0, 0)),
                    make_oi(make_ti(0, 2), make_ti(0, 0)),
                    make_oi(make_ti(0, 3), make_ti(0, 0))))
    return 1;
  // Identity: x + 0
  {
    oi_int big = make_oi(make_ti(0x1234, 0x5678), make_ti(0x9ABC, 0xDEF0));
    if (test__addvoi3(big, (oi_int)0, big))
      return 1;
  }
  // Additive inverse
  if (test__addvoi3(make_oi(make_ti(0, 1), make_ti(0, 0)),
                    make_oi(make_ti(-1, -1), make_ti(0, 0)), (oi_int)0))
    return 1;
  // Full-width big-number test (all 4 limbs populated).
  // Expected value verified by Python arbitrary-precision arithmetic.
  // A(signed) + B(signed) = 0xBBBBDDDE...99981111
  if (test__addvoi3(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          make_oi(make_ti(0x1111222233334444LL, 0x5555666677778888ULL),
                  make_ti(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          make_oi(make_ti(0xBBBBDDDE00002222LL, 0x444466658888AAAAULL),
                  make_ti(0xCCCCEEEF11113333ULL, 0x5555777799981111ULL))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
