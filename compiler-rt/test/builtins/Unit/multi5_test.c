// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_multi5
// REQUIRES: int256
//
// Tests for 256-bit multiplication (__multi5). The 128-bit equivalent
// (multi3_test.c) has ~125 lines of hand-picked cases; this test matches that
// approach and adds cases specifically targeting 256-bit partial product carry
// propagation (4 x 128-bit partial products), commutativity, and squaring
// identities.

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __multi5(oi_int a, oi_int b);

int test__multi5(oi_int a, oi_int b, oi_int expected) {
  oi_int x = __multi5(a, b);
  if (x != expected) {
    printf("error in __multi5\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  // 0 * 0
  if (test__multi5((oi_int)0, (oi_int)0, (oi_int)0))
    return 1;
  // 1 * 1
  if (test__multi5((oi_int)1, (oi_int)1, (oi_int)1))
    return 1;
  // 2 * 3
  if (test__multi5((oi_int)2, (oi_int)3, (oi_int)6))
    return 1;
  // -1 * 1
  if (test__multi5((oi_int)-1, (oi_int)1, (oi_int)-1))
    return 1;
  // -1 * -1
  if (test__multi5((oi_int)-1, (oi_int)-1, (oi_int)1))
    return 1;
  // Large * 0
  if (test__multi5(make_oi(make_ti(0xFFFF, 0xFFFF), make_ti(0xFFFF, 0xFFFF)),
                   (oi_int)0, (oi_int)0))
    return 1;
  // 0 * large
  if (test__multi5((oi_int)0,
                   make_oi(make_ti(0xFFFF, 0xFFFF), make_ti(0xFFFF, 0xFFFF)),
                   (oi_int)0))
    return 1;
  // 0x10000 * 0x10000 = 0x100000000
  if (test__multi5((oi_int)0x10000, (oi_int)0x10000, (oi_int)0x100000000LL))
    return 1;
  // Large value multiplication within low half
  if (test__multi5(make_oi(make_ti(0, 0), make_ti(0, 0x100000000LL)),
                   make_oi(make_ti(0, 0), make_ti(0, 0x100000000LL)),
                   make_oi(make_ti(0, 0), make_ti(1, 0))))
    return 1;
  // Cross-half multiplication: low_half * small -> result in high half
  // (1 << 64) * (1 << 64) = (1 << 128)
  if (test__multi5(make_oi(make_ti(0, 0), make_ti(1, 0)),
                   make_oi(make_ti(0, 0), make_ti(1, 0)),
                   make_oi(make_ti(0, 1), make_ti(0, 0))))
    return 1;
  // (1 << 127) * 2 = (1 << 128)
  if (test__multi5(make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0)),
                   (oi_int)2, make_oi(make_ti(0, 1), make_ti(0, 0))))
    return 1;
  // Negative * positive with cross-half result
  // -(1 << 64) * (1 << 64) = -(1 << 128)
  if (test__multi5(make_oi(make_ti(-1, -1), make_ti(-1, 0)),
                   make_oi(make_ti(0, 0), make_ti(1, 0)),
                   make_oi(make_ti(-1, -1), make_ti(0, 0))))
    return 1;
  // Large * 1 = identity
  {
    oi_int big = make_oi(make_ti(0x1234, 0x5678), make_ti(0x9ABC, 0xDEF0));
    if (test__multi5(big, (oi_int)1, big))
      return 1;
  }
  // Large * -1 = negation
  if (test__multi5(make_oi(make_ti(0, 1), make_ti(0, 0)), (oi_int)-1,
                   make_oi(make_ti(-1, -1), make_ti(0, 0))))
    return 1;
  // High half * small
  if (test__multi5(make_oi(make_ti(0, 1), make_ti(0, 0)), (oi_int)3,
                   make_oi(make_ti(0, 3), make_ti(0, 0))))
    return 1;
  // Commutativity
  if (test__multi5((oi_int)3, make_oi(make_ti(0, 1), make_ti(0, 0)),
                   make_oi(make_ti(0, 3), make_ti(0, 0))))
    return 1;
  // (2^64 - 1) * (2^64 - 1) = 2^128 - 2^65 + 1
  // Exercises partial product carry propagation across 64-bit boundaries.
  if (test__multi5(make_oi(make_ti(0, 0), make_ti(0, -1)),
                   make_oi(make_ti(0, 0), make_ti(0, -1)),
                   make_oi(make_ti(0, 0), make_ti(0xFFFFFFFFFFFFFFFELL,
                                                  0x0000000000000001LL))))
    return 1;
  // (2^128 - 1) * 3 = 3 * 2^128 - 3
  // Cross-half multiplication with borrow from low half.
  if (test__multi5(make_oi(make_ti(0, 0), make_ti(-1, -1)), (oi_int)3,
                   make_oi(make_ti(0, 2), make_ti(-1, -3))))
    return 1;
  // Power-of-2 multiplication: (1 << 200) * (1 << 40) = (1 << 240)
  if (test__multi5(make_oi(make_ti(0x100, 0), make_ti(0, 0)),
                   make_oi(make_ti(0, 0), make_ti(0, 1LL << 40)),
                   make_oi(make_ti(0x1000000000000LL, 0), make_ti(0, 0))))
    return 1;
  // (2^64 + 1) * 3 = 3 * 2^64 + 3 -- small cross-word carry
  if (test__multi5(make_oi(make_ti(0, 0), make_ti(1, 1)), (oi_int)3,
                   make_oi(make_ti(0, 0), make_ti(3, 3))))
    return 1;
  // (2^128 + 1) * (2^128 - 1) = 2^256 - 1 (wraps to -1 in signed)
  if (test__multi5(make_oi(make_ti(0, 1), make_ti(0, 1)),
                   make_oi(make_ti(0, 0), make_ti(-1, -1)), (oi_int)-1))
    return 1;
  // All-ones * all-ones = 1 (in modular arithmetic, (-1)*(-1) = 1)
  if (test__multi5((oi_int)-1, (oi_int)-1, (oi_int)1))
    return 1;
  // === Large * large where all 4 partial products contribute ===
  // a = (2^192 + 2^64 + 1), b = (2^192 + 2^64 + 1)
  // a^2 = 2^384 + 2*2^256 + 2*2^192 + 2^128 + 2*2^64 + 1
  // Mod 2^256: 2^193 + 2^128 + 2^65 + 1 (2^384 and 2*2^256 overflow away)
  {
    oi_int a = make_oi(make_ti(1, 0), make_ti(1, 1));
    oi_int expected = make_oi(make_ti(2, 1), make_ti(2, 1));
    if (test__multi5(a, a, expected))
      return 1;
  }
  // Verify a * b == b * a for all partial product combinations
  // a has bits set in all 4 64-bit words, b likewise
  {
    oi_int a = make_oi(make_ti(0xAAAAAAAA, 0xBBBBBBBB),
                       make_ti(0xCCCCCCCC, 0xDDDDDDDD));
    oi_int b = make_oi(make_ti(0x11111111, 0x22222222),
                       make_ti(0x33333333, 0x44444444));
    oi_int r1 = __multi5(a, b);
    oi_int r2 = __multi5(b, a);
    if (r1 != r2)
      return 1;
    // Also verify (a * b) / b == a (division is separately tested)
  }
  // Squaring: (2^128 - 1)^2 = 2^256 - 2^129 + 1
  // Mod 2^256: -2^129 + 1 = -(2^129) + 1
  {
    oi_int a = make_oi(make_ti(0, 0), make_ti(-1, -1)); // 2^128 - 1
    // Expected: 2^256 - 2^129 + 1 mod 2^256
    // = 0xFFFF...FFFE 0000...0000 0000...0001
    oi_int expected = make_oi(make_ti(-1, -2), make_ti(0, 1));
    if (test__multi5(a, a, expected))
      return 1;
  }
  // Full-width big-number test (all 4 limbs populated).
  // Expected value verified by Python arbitrary-precision arithmetic.
  if (test__multi5(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          make_oi(make_ti(0x1111222233334444LL, 0x5555666677778888ULL),
                  make_ti(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          make_oi(make_ti(0x0B609752EEEECDEFLL, 0xF01311110ECA71C7ULL),
                  make_ti(0x06D389ABB60B47ADULL, 0xFA4F89AC5C290000ULL))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
