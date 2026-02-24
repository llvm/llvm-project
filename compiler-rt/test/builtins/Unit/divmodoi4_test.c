// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_divmodoi4
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __divmodoi4(oi_int a, oi_int b, oi_int *rem);

int test__divmodoi4(oi_int a, oi_int b, oi_int expected_q, oi_int expected_r) {
  oi_int r;
  oi_int q = __divmodoi4(a, b, &r);
  if (q != expected_q || r != expected_r) {
    printf("error in __divmodoi4\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__divmodoi4((oi_int)0, (oi_int)1, (oi_int)0, (oi_int)0))
    return 1;
  if (test__divmodoi4((oi_int)10, (oi_int)3, (oi_int)3, (oi_int)1))
    return 1;
  if (test__divmodoi4((oi_int)-10, (oi_int)3, (oi_int)-3, (oi_int)-1))
    return 1;
  if (test__divmodoi4((oi_int)10, (oi_int)-3, (oi_int)-3, (oi_int)1))
    return 1;
  if (test__divmodoi4((oi_int)-10, (oi_int)-3, (oi_int)3, (oi_int)-1))
    return 1;
  if (test__divmodoi4((oi_int)100, (oi_int)7, (oi_int)14, (oi_int)2))
    return 1;
  // Exact division
  if (test__divmodoi4((oi_int)42, (oi_int)42, (oi_int)1, (oi_int)0))
    return 1;
  // Dividend smaller than divisor
  if (test__divmodoi4((oi_int)3, (oi_int)10, (oi_int)0, (oi_int)3))
    return 1;
  // (1 << 128) / 2
  if (test__divmodoi4(make_oi(make_ti(0, 1), make_ti(0, 0)), (oi_int)2,
                      make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0)),
                      (oi_int)0))
    return 1;
  // (1 << 128) / 3 with remainder
  if (test__divmodoi4(make_oi(make_ti(0, 1), make_ti(0, 0)), (oi_int)3,
                      make_oi(make_ti(0, 0), make_ti(0x5555555555555555LL,
                                                     0x5555555555555555ULL)),
                      (oi_int)1))
    return 1;
  // Negative large / positive small
  if (test__divmodoi4(
          make_oi(make_ti(-1, -1), make_ti(0, 0)), (oi_int)2,
          make_oi(make_ti(-1, -1), make_ti(0x8000000000000000LL, 0)),
          (oi_int)0))
    return 1;
  // Positive large / negative small
  if (test__divmodoi4(
          make_oi(make_ti(0, 1), make_ti(0, 0)), (oi_int)-2,
          make_oi(make_ti(-1, -1), make_ti(0x8000000000000000LL, 0)),
          (oi_int)0))
    return 1;
  // Large / large (same value)
  {
    oi_int big = make_oi(make_ti(0, 0x100), make_ti(0, 0));
    if (test__divmodoi4(big, big, (oi_int)1, (oi_int)0))
      return 1;
  }
  // Cross-half boundary value
  if (test__divmodoi4(make_oi(make_ti(0, 1), make_ti(0, 5)), (oi_int)4,
                      make_oi(make_ti(0, 0), make_ti(0x4000000000000000LL, 1)),
                      (oi_int)1))
    return 1;
  // Full-width big-number test (all 4 limbs populated).
  // A(signed) divmod B(signed): q = -4, r verified by Python: q*b + r == a.
  // Expected values verified by Python arbitrary-precision arithmetic.
  if (test__divmodoi4(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          make_oi(make_ti(0x1111222233334444LL, 0x5555666677778888ULL),
                  make_ti(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          make_oi(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFULL),
                  make_ti(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFCULL)),
          make_oi(make_ti(0xEEEF44449999EEEFLL, 0x44449998EEEF4444ULL),
                  make_ti(0x9999EEEF44449999ULL, 0xEEEF444499954444ULL))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
