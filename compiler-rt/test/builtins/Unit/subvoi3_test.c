// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_subvoi3
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __subvoi3(oi_int a, oi_int b);

int test__subvoi3(oi_int a, oi_int b, oi_int expected) {
  oi_int x = __subvoi3(a, b);
  if (x != expected) {
    printf("error in __subvoi3\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__subvoi3((oi_int)0, (oi_int)0, (oi_int)0))
    return 1;
  if (test__subvoi3((oi_int)2, (oi_int)1, (oi_int)1))
    return 1;
  if (test__subvoi3((oi_int)0, (oi_int)1, (oi_int)-1))
    return 1;
  if (test__subvoi3((oi_int)300, (oi_int)200, (oi_int)100))
    return 1;
  // Negative result
  if (test__subvoi3((oi_int)100, (oi_int)200, (oi_int)-100))
    return 1;
  // Negative - positive
  if (test__subvoi3((oi_int)-100, (oi_int)200, (oi_int)-300))
    return 1;
  // Negative - negative
  if (test__subvoi3((oi_int)-100, (oi_int)-200, (oi_int)100))
    return 1;
  // Borrow across 128-bit boundary (high half to low half)
  if (test__subvoi3(make_oi(make_ti(0, 1), make_ti(0, 0)),
                    make_oi(make_ti(0, 0), make_ti(0, 1)),
                    make_oi(make_ti(0, 0), make_ti(-1, -1))))
    return 1;
  // Large values
  if (test__subvoi3(make_oi(make_ti(0, 3), make_ti(0, 0)),
                    make_oi(make_ti(0, 1), make_ti(0, 0)),
                    make_oi(make_ti(0, 2), make_ti(0, 0))))
    return 1;
  // x - x = 0
  {
    oi_int big = make_oi(make_ti(0x1234, 0x5678), make_ti(0x9ABC, 0xDEF0));
    if (test__subvoi3(big, big, (oi_int)0))
      return 1;
  }
  // x - 0 = x
  {
    oi_int big = make_oi(make_ti(0x1234, 0x5678), make_ti(0x9ABC, 0xDEF0));
    if (test__subvoi3(big, (oi_int)0, big))
      return 1;
  }
  // Full-width big-number test (all 4 limbs populated).
  // Expected value verified by Python arbitrary-precision arithmetic.
  // B(signed) - A(signed) = 0x66666666...66661111
  if (test__subvoi3(
          make_oi(make_ti(0x1111222233334444LL, 0x5555666677778888ULL),
                  make_ti(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          make_oi(make_ti(0x6666666666666666LL, 0x6666666766666666ULL),
                  make_ti(0x6666666666666666ULL, 0x6666666666661111ULL))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
