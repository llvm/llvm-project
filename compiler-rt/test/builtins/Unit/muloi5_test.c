// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_muloi5
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __muloi5(oi_int a, oi_int b, int *overflow);

int test__muloi5(oi_int a, oi_int b, oi_int expected, int expected_overflow) {
  int overflow;
  oi_int x = __muloi5(a, b, &overflow);
  if (overflow != expected_overflow || (!expected_overflow && x != expected)) {
    printf("error in __muloi5: overflow=%d (expected %d)\n", overflow,
           expected_overflow);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__muloi5((oi_int)0, (oi_int)0, (oi_int)0, 0))
    return 1;
  if (test__muloi5((oi_int)1, (oi_int)1, (oi_int)1, 0))
    return 1;
  if (test__muloi5((oi_int)2, (oi_int)3, (oi_int)6, 0))
    return 1;
  if (test__muloi5((oi_int)-1, (oi_int)1, (oi_int)-1, 0))
    return 1;
  if (test__muloi5((oi_int)-1, (oi_int)-1, (oi_int)1, 0))
    return 1;
  // Large * 0 = 0, no overflow
  if (test__muloi5(make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1)),
                   (oi_int)0, (oi_int)0, 0))
    return 1;
  // 0 * large = 0, no overflow
  if (test__muloi5((oi_int)0,
                   make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1)),
                   (oi_int)0, 0))
    return 1;
  // Cross-half multiplication without overflow
  // (1 << 64) * (1 << 64) = (1 << 128)
  if (test__muloi5(make_oi(make_ti(0, 0), make_ti(1, 0)),
                   make_oi(make_ti(0, 0), make_ti(1, 0)),
                   make_oi(make_ti(0, 1), make_ti(0, 0)), 0))
    return 1;
  // (1 << 127) * 2 = (1 << 128), no overflow
  if (test__muloi5(make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0)),
                   (oi_int)2, make_oi(make_ti(0, 1), make_ti(0, 0)), 0))
    return 1;
  // MAX * 1 = MAX, no overflow
  {
    oi_int MAX = make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1));
    if (test__muloi5(MAX, (oi_int)1, MAX, 0))
      return 1;
  }
  // MAX * 2 overflows
  {
    oi_int MAX = make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1));
    if (test__muloi5(MAX, (oi_int)2, (oi_int)0, 1))
      return 1;
  }
  // MIN * -1 overflows
  {
    oi_int MIN = make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0));
    if (test__muloi5(MIN, (oi_int)-1, (oi_int)0, 1))
      return 1;
  }
  // MIN * 1 = MIN, no overflow
  {
    oi_int MIN = make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0));
    if (test__muloi5(MIN, (oi_int)1, MIN, 0))
      return 1;
  }
  // (1 << 128) * (1 << 128) overflows (result would be 1 << 256)
  if (test__muloi5(make_oi(make_ti(0, 1), make_ti(0, 0)),
                   make_oi(make_ti(0, 1), make_ti(0, 0)), (oi_int)0, 1))
    return 1;
  // Negative * negative = positive, no overflow
  if (test__muloi5((oi_int)-100, (oi_int)-200, (oi_int)20000, 0))
    return 1;
  // === Near-overflow boundary tests ===
  {
    oi_int MAX = make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -1));
    oi_int MIN = make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0));
    // MAX / 2 * 2 = MAX - 1 (since MAX is odd), no overflow
    oi_int half_max = MAX >> 1; // = (MAX-1)/2
    if (test__muloi5(
            half_max, (oi_int)2,
            make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1), make_ti(-1, -2)), 0))
      return 1;
    // (MAX/2 + 1) * 2 = MAX + 1, overflows
    if (test__muloi5(half_max + 1, (oi_int)2, (oi_int)0, 1))
      return 1;
    // MIN / 2 * 2 = MIN, no overflow
    oi_int half_min = MIN >> 1; // = MIN/2
    if (test__muloi5(half_min, (oi_int)2, MIN, 0))
      return 1;
    // (MIN/2 - 1) * 2 = MIN - 2, overflows
    if (test__muloi5(half_min - 1, (oi_int)2, (oi_int)0, 1))
      return 1;
    // MAX * -1 = -MAX (= MIN + 1), no overflow
    if (test__muloi5(MAX, (oi_int)-1,
                     make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 1)),
                     0))
      return 1;
    // MIN * 2 overflows
    if (test__muloi5(MIN, (oi_int)2, (oi_int)0, 1))
      return 1;
    // MAX * -2 overflows
    if (test__muloi5(MAX, (oi_int)-2, (oi_int)0, 1))
      return 1;
    // (1 << 127) * (1 << 127) = (1 << 254), no overflow
    if (test__muloi5(make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0)),
                     make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0)),
                     make_oi(make_ti(0x4000000000000000LL, 0), make_ti(0, 0)),
                     0))
      return 1;
    // (1 << 128) * (1 << 126) = (1 << 254), no overflow
    if (test__muloi5(make_oi(make_ti(0, 1), make_ti(0, 0)),
                     make_oi(make_ti(0, 0), make_ti(0x4000000000000000LL, 0)),
                     make_oi(make_ti(0x4000000000000000LL, 0), make_ti(0, 0)),
                     0))
      return 1;
    // (1 << 128) * (1 << 127) = (1 << 255) overflows (== MIN as unsigned,
    // but as signed this is negative and the operands are both positive)
    if (test__muloi5(make_oi(make_ti(0, 1), make_ti(0, 0)),
                     make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0)),
                     (oi_int)0, 1))
      return 1;
  }
  // === Commutativity check ===
  {
    int ov1, ov2;
    oi_int a = make_oi(make_ti(0x12345678LL, 0), make_ti(0, 0xABCDEF01ULL));
    oi_int b = make_oi(make_ti(0, 0), make_ti(0, 0xFEDCBA98ULL));
    oi_int r1 = __muloi5(a, b, &ov1);
    oi_int r2 = __muloi5(b, a, &ov2);
    if (r1 != r2 || ov1 != ov2)
      return 1;
  }
  // Full-width big-number multiplication (fits in 255 bits, no overflow).
  // Expected value verified by Python arbitrary-precision arithmetic.
  if (test__muloi5(
          make_oi(make_ti(0x0000000000000000LL, 0x0000000000000000ULL),
                  make_ti(0x7766554433221100ULL, 0xFFEEDDCCBBAA9988ULL)),
          make_oi(make_ti(0x0000000000000000LL, 0x0000000000000000ULL),
                  make_ti(0x0000000000000002ULL, 0x1111111111111111ULL)),
          make_oi(make_ti(0x0000000000000000LL, 0xF6C26BF3589BBCBDULL),
                  make_ti(0xC4B3A291806F5E4CULL, 0x3334579D048E3A08ULL)),
          0))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
