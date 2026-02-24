// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_divoi3
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __divoi3(oi_int a, oi_int b);

int test__divoi3(oi_int a, oi_int b, oi_int expected) {
  oi_int x = __divoi3(a, b);
  if (x != expected) {
    printf("error in __divoi3\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__divoi3((oi_int)0, (oi_int)1, (oi_int)0))
    return 1;
  if (test__divoi3((oi_int)10, (oi_int)3, (oi_int)3))
    return 1;
  if (test__divoi3((oi_int)-10, (oi_int)3, (oi_int)-3))
    return 1;
  if (test__divoi3((oi_int)10, (oi_int)-3, (oi_int)-3))
    return 1;
  if (test__divoi3((oi_int)-10, (oi_int)-3, (oi_int)3))
    return 1;
  if (test__divoi3((oi_int)1, (oi_int)1, (oi_int)1))
    return 1;
  if (test__divoi3((oi_int)100, (oi_int)10, (oi_int)10))
    return 1;
  // Large dividend in high half / small divisor
  // (1 << 128) / 2 = (1 << 127)
  if (test__divoi3(make_oi(make_ti(0, 1), make_ti(0, 0)), (oi_int)2,
                   make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0))))
    return 1;
  // (1 << 128) / 3
  if (test__divoi3(make_oi(make_ti(0, 1), make_ti(0, 0)), (oi_int)3,
                   make_oi(make_ti(0, 0), make_ti(0x5555555555555555LL,
                                                  0x5555555555555555ULL))))
    return 1;
  // Negative large dividend
  // -(1 << 128) / 2 = -(1 << 127)
  if (test__divoi3(make_oi(make_ti(-1, -1), make_ti(0, 0)), (oi_int)2,
                   make_oi(make_ti(-1, -1), make_ti(0x8000000000000000LL, 0))))
    return 1;
  // Large / large (same value)
  {
    oi_int big = make_oi(make_ti(0, 0x100), make_ti(0, 0));
    if (test__divoi3(big, big, (oi_int)1))
      return 1;
  }
  // Large / large (double)
  {
    oi_int big = make_oi(make_ti(0, 0x100), make_ti(0, 0));
    oi_int dbl = make_oi(make_ti(0, 0x200), make_ti(0, 0));
    if (test__divoi3(dbl, big, (oi_int)2))
      return 1;
  }
  // Dividend smaller than divisor
  if (test__divoi3((oi_int)3, (oi_int)10, (oi_int)0))
    return 1;
  // Large negative / large negative
  {
    oi_int neg = make_oi(make_ti(-1, -2), make_ti(0, 0));
    if (test__divoi3(neg, neg, (oi_int)1))
      return 1;
  }
  // Cross-half boundary: value spans both halves
  if (test__divoi3(make_oi(make_ti(0, 1), make_ti(0, 4)), (oi_int)4,
                   make_oi(make_ti(0, 0), make_ti(0x4000000000000000LL, 1))))
    return 1;
  // Full-width big-number test (all 4 limbs populated).
  // A(signed) / B(signed) = -4 (truncation toward zero).
  // Expected value verified by Python arbitrary-precision arithmetic.
  if (test__divoi3(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          make_oi(make_ti(0x1111222233334444LL, 0x5555666677778888ULL),
                  make_ti(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          make_oi(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFULL),
                  make_ti(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFCULL))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
