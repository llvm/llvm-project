// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_ashroi3
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __ashroi3(oi_int a, int b);

int test__ashroi3(oi_int a, int b, oi_int expected) {
  oi_int x = __ashroi3(a, b);
  if (x != expected) {
    printf("error in __ashroi3: shift by %d\n", b);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  // Shift by 0
  if (test__ashroi3((oi_int)1, 0, (oi_int)1))
    return 1;
  // Shift positive by small amounts
  if (test__ashroi3((oi_int)2, 1, (oi_int)1))
    return 1;
  if (test__ashroi3((oi_int)4, 2, (oi_int)1))
    return 1;
  // Shift negative by 1 (sign extension)
  if (test__ashroi3((oi_int)-2, 1, (oi_int)-1))
    return 1;
  // Shift -1 by any amount stays -1 (sign extension)
  if (test__ashroi3((oi_int)-1, 1, (oi_int)-1))
    return 1;
  if (test__ashroi3((oi_int)-1, 64, (oi_int)-1))
    return 1;
  if (test__ashroi3((oi_int)-1, 128, (oi_int)-1))
    return 1;
  if (test__ashroi3((oi_int)-1, 255, (oi_int)-1))
    return 1;
  // Shift by 64 (within low half)
  if (test__ashroi3(make_oi(make_ti(0, 0), make_ti(0xABCD000000000000LL, 0)),
                    64,
                    make_oi(make_ti(0, 0), make_ti(0, 0xABCD000000000000ULL))))
    return 1;
  // Shift by 128 (crosses half boundary, positive)
  if (test__ashroi3(make_oi(make_ti(0, 1), make_ti(0, 0)), 128, (oi_int)1))
    return 1;
  // Shift by 128 (negative, sign extends)
  if (test__ashroi3(make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0)),
                    128,
                    make_oi(make_ti(-1, -1), make_ti(0x8000000000000000LL, 0))))
    return 1;
  // Shift by 192
  if (test__ashroi3(make_oi(make_ti(0x0000ABCD00000000LL, 0), make_ti(0, 0)),
                    192, (oi_int)0x0000ABCD00000000LL))
    return 1;
  // Shift MSB-only by 255
  if (test__ashroi3(make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0)),
                    255, (oi_int)-1))
    return 1;
  // Shift MAX positive by 255
  if (test__ashroi3(make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1)),
                    255, (oi_int)0))
    return 1;
  // Full-width big-number test (negative value, shift crosses 64-bit boundary).
  // A is negative in signed interpretation; arithmetic shift sign-extends.
  // Expected value verified by Python arbitrary-precision arithmetic.
  if (test__ashroi3(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          73,
          make_oi(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFD5555DDDE6666EULL),
                  make_ti(0xEEF7777FFF888891ULL, 0x111999A2222AAAB3ULL))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
