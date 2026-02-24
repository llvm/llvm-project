// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_lshroi3
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __lshroi3(oi_int a, int b);

int test__lshroi3(oi_int a, int b, oi_int expected) {
  oi_int x = __lshroi3(a, b);
  if (x != expected) {
    printf("error in __lshroi3: shift by %d\n", b);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  // Shift by 0
  if (test__lshroi3((oi_int)1, 0, (oi_int)1))
    return 1;
  // Shift by 1
  if (test__lshroi3((oi_int)2, 1, (oi_int)1))
    return 1;
  // Logical shift negative by 1 (no sign extension)
  if (test__lshroi3(
          (oi_int)-1, 1,
          make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1))))
    return 1;
  // Shift by 63
  if (test__lshroi3(make_oi(make_ti(0, 0), make_ti(0, 0x8000000000000000ULL)),
                    63, (oi_int)1))
    return 1;
  // Shift by 64
  if (test__lshroi3(make_oi(make_ti(0, 0), make_ti(1, 0)), 64, (oi_int)1))
    return 1;
  // Shift by 127
  if (test__lshroi3(make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0)),
                    127, (oi_int)1))
    return 1;
  // Shift by 128
  if (test__lshroi3(make_oi(make_ti(0, 1), make_ti(0, 0)), 128, (oi_int)1))
    return 1;
  // Shift by 129
  if (test__lshroi3(make_oi(make_ti(0, 2), make_ti(0, 0)), 129, (oi_int)1))
    return 1;
  // Shift by 191
  if (test__lshroi3(make_oi(make_ti(0, 0x8000000000000000ULL), make_ti(0, 0)),
                    191, (oi_int)1))
    return 1;
  // Shift by 192
  if (test__lshroi3(make_oi(make_ti(0xABCDLL, 0), make_ti(0, 0)), 192,
                    (oi_int)0xABCDLL))
    return 1;
  // Shift all-ones by 255
  if (test__lshroi3((oi_int)(ou_int)-1, 255, (oi_int)1))
    return 1;
  // Multi-bit value shift by 64
  if (test__lshroi3(make_oi(make_ti(0, 0), make_ti(0xFFFFFFFFFFFFFFFFULL, 0)),
                    64,
                    make_oi(make_ti(0, 0), make_ti(0, 0xFFFFFFFFFFFFFFFFULL))))
    return 1;
  // Multi-bit value shift by 128
  if (test__lshroi3(make_oi(make_ti(0, 0xFFFFFFFFFFFFFFFFULL), make_ti(0, 0)),
                    128,
                    make_oi(make_ti(0, 0), make_ti(0, 0xFFFFFFFFFFFFFFFFULL))))
    return 1;
  // Shift that spans both halves
  if (test__lshroi3(make_oi(make_ti(0, 1), make_ti(0, 0)), 1,
                    make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0))))
    return 1;
  // Full value shift by 0 (identity)
  if (test__lshroi3(
          make_oi(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL),
                  make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL)),
          0,
          make_oi(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL),
                  make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL))))
    return 1;
  // Full-width big-number test (all 4 limbs populated, shift crosses 64-bit boundary).
  // Expected value verified by Python arbitrary-precision arithmetic.
  if (test__lshroi3(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          73,
          make_oi(make_ti(0x0000000000000000LL, 0x0055555DDDE6666EULL),
                  make_ti(0xEEF7777FFF888891ULL, 0x111999A2222AAAB3ULL))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
