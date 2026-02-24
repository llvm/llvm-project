// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_udivmodoi4
// REQUIRES: int256
//
// Testing strategy: The 128-bit equivalent (udivmodti4_test.c) uses a 65K-line
// auto-generated exhaustive test vector file from the initial compiler-rt
// import (no generator script available). Instead of replicating that approach
// for 256-bit, this test uses:
//   1. Hand-picked edge cases covering both code paths in the Knuth algorithm
//      (Path 1: divisor fits in 128 bits, Path 2: divisor spans both halves)
//   2. A 100-iteration pseudo-random invariant checker that verifies
//      q * b + r == a and r < b for diverse LCG-generated inputs
//   3. A divisor size sweep from 1-bit to 255-bit divisors
// This catches the same class of bugs as exhaustive enumeration while being
// maintainable and readable.

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI ou_int __udivmodoi4(ou_int a, ou_int b, ou_int *rem);

int test__udivmodoi4(ou_int a, ou_int b, ou_int expected_q, ou_int expected_r) {
  ou_int r;
  ou_int q = __udivmodoi4(a, b, &r);
  if (q != expected_q || r != expected_r) {
    printf("error in __udivmodoi4\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  // 0 / 1
  if (test__udivmodoi4((ou_int)0, (ou_int)1, (ou_int)0, (ou_int)0))
    return 1;
  // 1 / 1
  if (test__udivmodoi4((ou_int)1, (ou_int)1, (ou_int)1, (ou_int)0))
    return 1;
  // 10 / 3
  if (test__udivmodoi4((ou_int)10, (ou_int)3, (ou_int)3, (ou_int)1))
    return 1;
  // 100 / 7
  if (test__udivmodoi4((ou_int)100, (ou_int)7, (ou_int)14, (ou_int)2))
    return 1;
  // Large value / small
  if (test__udivmodoi4(
          make_ou(make_tu(0, 0), make_tu(1, 0)), (ou_int)2,
          make_ou(make_tu(0, 0), make_tu(0, 0x8000000000000000ULL)), (ou_int)0))
    return 1;
  // Dividend < divisor
  if (test__udivmodoi4((ou_int)3, (ou_int)10, (ou_int)0, (ou_int)3))
    return 1;
  // Equal
  if (test__udivmodoi4((ou_int)42, (ou_int)42, (ou_int)1, (ou_int)0))
    return 1;
  // Large divisor (both halves)
  {
    ou_int big = make_ou(make_tu(0, 1), make_tu(0, 0));
    if (test__udivmodoi4(big, big, (ou_int)1, (ou_int)0))
      return 1;
  }
  // (1 << 128) / 3 = quotient with remainder 1
  if (test__udivmodoi4(make_ou(make_tu(0, 1), make_tu(0, 0)), (ou_int)3,
                       make_ou(make_tu(0, 0), make_tu(0x5555555555555555ULL,
                                                      0x5555555555555555ULL)),
                       (ou_int)1))
    return 1;
  // All-ones / 2 = 0x7FFF...FFFF remainder 1
  if (test__udivmodoi4(
          (ou_int)-1, (ou_int)2,
          make_ou(make_tu(0x7FFFFFFFFFFFFFFFULL, -1), make_tu(-1, -1)),
          (ou_int)1))
    return 1;
  // Cross-half boundary: value spans both halves
  if (test__udivmodoi4(
          make_ou(make_tu(0, 1), make_tu(0, 5)), (ou_int)4,
          make_ou(make_tu(0, 0), make_tu(0x4000000000000000ULL, 1)), (ou_int)1))
    return 1;
  // Large / large (double)
  {
    ou_int big = make_ou(make_tu(0, 0x100), make_tu(0, 0));
    ou_int dbl = make_ou(make_tu(0, 0x200), make_tu(0, 0));
    if (test__udivmodoi4(dbl, big, (ou_int)2, (ou_int)0))
      return 1;
  }
  // Very large divisor in high half
  {
    ou_int big = make_ou(make_tu(1, 0), make_tu(0, 0));
    if (test__udivmodoi4(big, big, (ou_int)1, (ou_int)0))
      return 1;
  }
  // Large value with remainder
  {
    ou_int big = make_ou(make_tu(0, 0x100), make_tu(0, 7));
    ou_int div = make_ou(make_tu(0, 0x100), make_tu(0, 0));
    if (test__udivmodoi4(big, div, (ou_int)1, (ou_int)7))
      return 1;
  }
  // Division by power of 2 vs equivalent shift: (1 << 192) / (1 << 64)
  // = (1 << 128). Path 1: divisor.s.high == 0.
  if (test__udivmodoi4(make_ou(make_tu(1, 0), make_tu(0, 0)),
                       make_ou(make_tu(0, 0), make_tu(1, 0)),
                       make_ou(make_tu(0, 1), make_tu(0, 0)), (ou_int)0))
    return 1;
  // Path 1: Large dividend / medium 128-bit divisor.
  // (2^192 + 2^64) / (2^64) = 2^128 + 1, remainder 0.
  if (test__udivmodoi4(make_ou(make_tu(1, 0), make_tu(1, 0)),
                       make_ou(make_tu(0, 0), make_tu(1, 0)),
                       make_ou(make_tu(0, 1), make_tu(0, 1)), (ou_int)0))
    return 1;
  // Path 1: dividend.s.high >= divisor.s.low (needs two-step division).
  // (3 * 2^128) / (2^128 - 1) = 3, remainder 3.
  if (test__udivmodoi4(make_ou(make_tu(0, 3), make_tu(0, 0)),
                       make_ou(make_tu(0, 0), make_tu(-1, -1)), (ou_int)3,
                       (ou_int)3))
    return 1;
  // Path 2: Both halves set in divisor. Bit-by-bit division.
  // (2^256 - 1) / (2^128 + 1) = 2^128 - 1, remainder 0.
  if (test__udivmodoi4((ou_int)-1, make_ou(make_tu(0, 1), make_tu(0, 1)),
                       make_ou(make_tu(0, 0), make_tu(-1, -1)), (ou_int)0))
    return 1;
  // Path 2: Large 256-bit divisor with remainder.
  // (2^255) / (2^254 + 1): quotient = 1, remainder = 2^254 - 1.
  {
    ou_int dividend = make_ou(make_tu(0x8000000000000000ULL, 0), make_tu(0, 0));
    ou_int divisor = make_ou(make_tu(0x4000000000000000ULL, 0), make_tu(0, 1));
    ou_int exp_q = (ou_int)1;
    ou_int exp_r = make_ou(make_tu(0x3FFFFFFFFFFFFFFFULL, -1), make_tu(-1, -1));
    if (test__udivmodoi4(dividend, divisor, exp_q, exp_r))
      return 1;
  }
  // Verify q * b + r == a invariant for a non-trivial case.
  // a = 0xDEADBEEF12345678 (repeated), b = 0xCAFEBABE (fits in 128 bits).
  {
    ou_int a = make_ou(make_tu(0xDEADBEEF12345678ULL, 0xDEADBEEF12345678ULL),
                       make_tu(0xDEADBEEF12345678ULL, 0xDEADBEEF12345678ULL));
    ou_int b = (ou_int)0xCAFEBABEULL;
    ou_int r;
    ou_int q = __udivmodoi4(a, b, &r);
    if (q * b + r != a)
      return 1;
    // Remainder must be less than divisor.
    if (r >= b)
      return 1;
  }
  // Verify q * b + r == a for a large divisor spanning both halves.
  {
    ou_int a = make_ou(make_tu(0xAAAAAAAAAAAAAAAAULL, 0xBBBBBBBBBBBBBBBBULL),
                       make_tu(0xCCCCCCCCCCCCCCCCULL, 0xDDDDDDDDDDDDDDDDULL));
    ou_int b = make_ou(make_tu(0, 0x1234567890ABCDEFULL),
                       make_tu(0xFEDCBA0987654321ULL, 0x1111111111111111ULL));
    ou_int r;
    ou_int q = __udivmodoi4(a, b, &r);
    if (q * b + r != a)
      return 1;
    if (r >= b)
      return 1;
  }
  // Full-width big-number test (all 4 limbs populated).
  // A / B (unsigned): q = 9, r verified by Python: q*b + r == a.
  // Expected values verified by Python arbitrary-precision arithmetic.
  if (test__udivmodoi4(
          make_ou(make_tu(0xAAAABBBBCCCCDDDDULL, 0xEEEEFFFF11112222ULL),
                  make_tu(0x3333444455556666ULL, 0x7777888899990000ULL)),
          make_ou(make_tu(0x1111222233334444ULL, 0x5555666677778888ULL),
                  make_tu(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          (ou_int)9,
          make_ou(make_tu(0x11108887FFFF7776ULL, 0xEEEE6664DDDD5554ULL),
                  make_tu(0xCCCC4443BBBB3332ULL, 0xAAAA222199A16667ULL))))
    return 1;
  // === Pseudo-random invariant checker ===
  // Generate ~100 test vectors using a simple LCG and verify q * b + r == a
  // and r < b for each. This catches systematic bugs in the Knuth algorithm
  // that hand-picked cases might miss.
  {
    // LCG parameters (Numerical Recipes)
    unsigned long long seed = 0xDEADBEEFCAFEBABEULL;
    int failures = 0;
    for (int i = 0; i < 100; ++i) {
      // Generate pseudo-random a and b using LCG
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      unsigned long long w0 = seed;
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      unsigned long long w1 = seed;
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      unsigned long long w2 = seed;
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      unsigned long long w3 = seed;
      ou_int a = make_ou(make_tu(w3, w2), make_tu(w1, w0));

      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      unsigned long long d0 = seed;
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      unsigned long long d1 = seed;
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      unsigned long long d2 = seed;
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      unsigned long long d3 = seed;
      ou_int b = make_ou(make_tu(d3, d2), make_tu(d1, d0));
      if (b == 0)
        b = 1;

      ou_int r;
      ou_int q = __udivmodoi4(a, b, &r);
      // Invariant: q * b + r == a
      if (q * b + r != a) {
        printf("invariant failure at i=%d: q * b + r != a\n", i);
        failures++;
      }
      // Invariant: r < b
      if (r >= b) {
        printf("invariant failure at i=%d: r >= b\n", i);
        failures++;
      }
    }
    if (failures)
      return 1;
  }
  // === Divisor size sweep ===
  // Test with divisors of varying sizes: 1 bit, 32 bits, 64 bits,
  // 128 bits, 192 bits, 255 bits. This exercises both Path 1
  // (divisor.high == 0) and Path 2 (both halves).
  {
    ou_int dividend = (ou_int)-1; // max value
    ou_int r;
    ou_int q;
    // 1-bit divisor
    q = __udivmodoi4(dividend, (ou_int)1, &r);
    if (q != dividend || r != 0)
      return 1;
    // 32-bit divisor
    q = __udivmodoi4(dividend, (ou_int)0xFFFFFFFFULL, &r);
    if (q * (ou_int)0xFFFFFFFFULL + r != dividend)
      return 1;
    // 64-bit divisor
    q = __udivmodoi4(dividend, (ou_int)0xFFFFFFFFFFFFFFFFULL, &r);
    if (q * (ou_int)0xFFFFFFFFFFFFFFFFULL + r != dividend)
      return 1;
    // 128-bit divisor (all ones in low half)
    {
      ou_int d128 = make_ou(make_tu(0, 0), make_tu(-1, -1));
      q = __udivmodoi4(dividend, d128, &r);
      if (q * d128 + r != dividend)
        return 1;
    }
    // 192-bit divisor
    {
      ou_int d192 = make_ou(make_tu(0, -1), make_tu(-1, -1));
      q = __udivmodoi4(dividend, d192, &r);
      if (q * d192 + r != dividend)
        return 1;
    }
    // 255-bit divisor (max >> 1)
    {
      ou_int d255 = (ou_int)-1 >> 1;
      q = __udivmodoi4(dividend, d255, &r);
      if (q * d255 + r != dividend)
        return 1;
    }
  }
#else
  printf("skipped\n");
#endif
  return 0;
}
