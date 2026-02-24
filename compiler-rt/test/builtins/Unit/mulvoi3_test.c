// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_mulvoi3
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __mulvoi3(oi_int a, oi_int b);

int test__mulvoi3(oi_int a, oi_int b, oi_int expected) {
  oi_int x = __mulvoi3(a, b);
  if (x != expected) {
    printf("error in __mulvoi3\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__mulvoi3((oi_int)0, (oi_int)0, (oi_int)0))
    return 1;
  if (test__mulvoi3((oi_int)1, (oi_int)1, (oi_int)1))
    return 1;
  if (test__mulvoi3((oi_int)2, (oi_int)3, (oi_int)6))
    return 1;
  if (test__mulvoi3((oi_int)-1, (oi_int)1, (oi_int)-1))
    return 1;
  if (test__mulvoi3((oi_int)-1, (oi_int)-1, (oi_int)1))
    return 1;
  if (test__mulvoi3((oi_int)0x10000, (oi_int)0x10000, (oi_int)0x100000000LL))
    return 1;
  // Large * 0 = 0
  if (test__mulvoi3(make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1)),
                    (oi_int)0, (oi_int)0))
    return 1;
  // Cross-half multiplication: (1 << 64) * (1 << 64) = (1 << 128)
  if (test__mulvoi3(make_oi(make_ti(0, 0), make_ti(1, 0)),
                    make_oi(make_ti(0, 0), make_ti(1, 0)),
                    make_oi(make_ti(0, 1), make_ti(0, 0))))
    return 1;
  // Negative * positive
  if (test__mulvoi3((oi_int)-100, (oi_int)200, (oi_int)-20000))
    return 1;
  // Negative * negative
  if (test__mulvoi3((oi_int)-100, (oi_int)-200, (oi_int)20000))
    return 1;
  // Large * 1 = identity
  {
    oi_int big = make_oi(make_ti(0x1234, 0x5678), make_ti(0x9ABC, 0xDEF0));
    if (test__mulvoi3(big, (oi_int)1, big))
      return 1;
  }
  // Note: overflow cases would abort, so we don't test them.
  // Instead, we test the maximum non-overflowing products.

  // MAX * 1 = MAX
  {
    oi_int MAX = make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1));
    if (test__mulvoi3(MAX, (oi_int)1, MAX))
      return 1;
  }
  // MAX * -1 = -MAX (= MIN + 1)
  {
    oi_int MAX = make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1));
    oi_int NEG_MAX = make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 1));
    if (test__mulvoi3(MAX, (oi_int)-1, NEG_MAX))
      return 1;
  }
  // MIN * 1 = MIN
  {
    oi_int MIN = make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0));
    if (test__mulvoi3(MIN, (oi_int)1, MIN))
      return 1;
  }
  // (MAX/2) * 2 = MAX - 1 (MAX is odd)
  {
    oi_int MAX = make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1));
    oi_int half = MAX >> 1;
    oi_int expected =
        make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -2));
    if (test__mulvoi3(half, (oi_int)2, expected))
      return 1;
  }
  // (1 << 127) * (1 << 127) = (1 << 254), near MAX but not overflow
  if (test__mulvoi3(make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0)),
                    make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0)),
                    make_oi(make_ti(0x4000000000000000LL, 0), make_ti(0, 0))))
    return 1;
  // Commutativity
  if (test__mulvoi3((oi_int)17, (oi_int)19, (oi_int)323))
    return 1;
  if (test__mulvoi3((oi_int)19, (oi_int)17, (oi_int)323))
    return 1;
  // Large negative * negative = positive
  if (test__mulvoi3(make_oi(make_ti(-1, -1), make_ti(-1, -100)), (oi_int)-1,
                    make_oi(make_ti(0, 0), make_ti(0, 100))))
    return 1;
  // Full-width big-number multiplication (fits in 255 bits, no overflow).
  // Expected value verified by Python arbitrary-precision arithmetic.
  if (test__mulvoi3(
          make_oi(make_ti(0x0000000000000000LL, 0x0000000000000000ULL),
                  make_ti(0x7766554433221100ULL, 0xFFEEDDCCBBAA9988ULL)),
          make_oi(make_ti(0x0000000000000000LL, 0x0000000000000000ULL),
                  make_ti(0x0000000000000002ULL, 0x1111111111111111ULL)),
          make_oi(make_ti(0x0000000000000000LL, 0xF6C26BF3589BBCBDULL),
                  make_ti(0xC4B3A291806F5E4CULL, 0x3334579D048E3A08ULL))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
