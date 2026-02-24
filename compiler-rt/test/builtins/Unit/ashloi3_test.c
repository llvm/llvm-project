// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_ashloi3
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

// Returns: a << b

// Precondition:  0 <= b < bits_in_oword

COMPILER_RT_ABI oi_int __ashloi3(oi_int a, int b);

int test__ashloi3(oi_int a, int b, oi_int expected) {
  oi_int x = __ashloi3(a, b);
  if (x != expected) {
    owords xt;
    xt.all = x;
    owords expectedt;
    expectedt.all = expected;
    printf("error in __ashloi3: shift by %d\n", b);
    printf("  got:      0x%.16llX%.16llX%.16llX%.16llX\n",
           (unsigned long long)((tu_int)xt.s.high >> 64),
           (unsigned long long)xt.s.high,
           (unsigned long long)((tu_int)xt.s.low >> 64),
           (unsigned long long)xt.s.low);
    printf("  expected: 0x%.16llX%.16llX%.16llX%.16llX\n",
           (unsigned long long)((tu_int)expectedt.s.high >> 64),
           (unsigned long long)expectedt.s.high,
           (unsigned long long)((tu_int)expectedt.s.low >> 64),
           (unsigned long long)expectedt.s.low);
  }
  return x != expected;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  // Shift by 0 (identity)
  if (test__ashloi3(
          make_oi(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL),
                  make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL)),
          0,
          make_oi(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL),
                  make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL))))
    return 1;
  // Shift by 1
  if (test__ashloi3((ou_int)1, 1, (ou_int)2))
    return 1;
  if (test__ashloi3((ou_int)1, 2, (ou_int)4))
    return 1;
  if (test__ashloi3((ou_int)1, 4, (ou_int)16))
    return 1;
  // Shift by 63 (within first 64-bit word)
  if (test__ashloi3((ou_int)1, 63,
                    make_oi(make_ti(0, 0), make_ti(0, 0x8000000000000000ULL))))
    return 1;
  // Shift by 64 (crosses into second 64-bit word)
  if (test__ashloi3((ou_int)1, 64, make_oi(make_ti(0, 0), make_ti(1, 0))))
    return 1;
  // Shift by 127 (top of low 128-bit half)
  if (test__ashloi3((ou_int)1, 127,
                    make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0))))
    return 1;
  // Shift by 128 (crosses into high 128-bit half)
  if (test__ashloi3((ou_int)1, 128, make_oi(make_ti(0, 1), make_ti(0, 0))))
    return 1;
  // Shift by 129
  if (test__ashloi3((ou_int)1, 129, make_oi(make_ti(0, 2), make_ti(0, 0))))
    return 1;
  // Shift by 191
  if (test__ashloi3((ou_int)1, 191,
                    make_oi(make_ti(0, 0x8000000000000000ULL), make_ti(0, 0))))
    return 1;
  // Shift by 192
  if (test__ashloi3((ou_int)1, 192, make_oi(make_ti(1, 0), make_ti(0, 0))))
    return 1;
  // Shift by 255 (MSB)
  if (test__ashloi3((ou_int)1, 255,
                    make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0))))
    return 1;
  // Multi-bit value shift by 64
  if (test__ashloi3((ou_int)0xFFFFFFFFFFFFFFFFULL, 64,
                    make_oi(make_ti(0, 0), make_ti(0xFFFFFFFFFFFFFFFFULL, 0))))
    return 1;
  // Multi-bit value shift by 128
  if (test__ashloi3((ou_int)0xFFFFFFFFFFFFFFFFULL, 128,
                    make_oi(make_ti(0, 0xFFFFFFFFFFFFFFFFULL), make_ti(0, 0))))
    return 1;
  // Multi-bit value shift by 192
  if (test__ashloi3((ou_int)0xFFFFFFFFFFFFFFFFULL, 192,
                    make_oi(make_ti(0xFFFFFFFFFFFFFFFFULL, 0), make_ti(0, 0))))
    return 1;
  // Full value shift crossing half boundary
  if (test__ashloi3(make_oi(make_ti(0, 0), make_ti(0, 0xABCDLL)), 4,
                    make_oi(make_ti(0, 0), make_ti(0, 0xABCD0LL))))
    return 1;
  // Shift that spans both halves
  if (test__ashloi3(make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL,
                                                   0x0000000000000001LL)),
                    1,
                    make_oi(make_ti(0, 1), make_ti(0, 0x0000000000000002LL))))
    return 1;
  // Full-width big-number test (all 4 limbs populated, shift crosses 64-bit boundary).
  // Expected value verified by Python arbitrary-precision arithmetic.
  if (test__ashloi3(
          make_oi(make_ti(0xAAAABBBBCCCCDDDDLL, 0xEEEEFFFF11112222ULL),
                  make_ti(0x3333444455556666ULL, 0x7777888899990000ULL)),
          73,
          make_oi(make_ti(0xDDFFFE2222444466LL, 0x668888AAAACCCCEEULL),
                  make_ti(0xEF11113332000000ULL, 0x0000000000000000ULL))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
