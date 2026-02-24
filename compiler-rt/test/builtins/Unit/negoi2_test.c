// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_negoi2
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __negoi2(oi_int a);

int test__negoi2(oi_int a, oi_int expected) {
  oi_int x = __negoi2(a);
  if (x != expected) {
    printf("error in __negoi2\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__negoi2((oi_int)0, (oi_int)0))
    return 1;
  if (test__negoi2((oi_int)1, (oi_int)-1))
    return 1;
  if (test__negoi2((oi_int)-1, (oi_int)1))
    return 1;
  if (test__negoi2((oi_int)42, (oi_int)-42))
    return 1;
  if (test__negoi2((oi_int)-42, (oi_int)42))
    return 1;
  // Large value in high half
  if (test__negoi2(make_oi(make_ti(0, 1), make_ti(0, 0)),
                   make_oi(make_ti(-1, -1), make_ti(0, 0))))
    return 1;
  // Negate back
  if (test__negoi2(make_oi(make_ti(-1, -1), make_ti(0, 0)),
                   make_oi(make_ti(0, 1), make_ti(0, 0))))
    return 1;
  // MAX
  if (test__negoi2(make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1)),
                   make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 1))))
    return 1;
  // Value with bits in low half only
  if (test__negoi2(make_oi(make_ti(0, 0), make_ti(0, 1)),
                   make_oi(make_ti(-1, -1), make_ti(-1, -1))))
    return 1;
  // Value spanning both halves
  if (test__negoi2(make_oi(make_ti(0, 1), make_ti(0, 1)),
                   make_oi(make_ti(-1, -2), make_ti(-1, -1))))
    return 1;
  // Full-width big-number test (all 4 limbs populated).
  // Expected value verified by Python arbitrary-precision arithmetic.
  if (test__negoi2(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          make_oi(make_ti(0x5555444433332222LL, 0x11110000EEEEDDDDULL),
                  make_ti(0xCCCCBBBBAAAA9999ULL, 0x8888777766670000ULL))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
