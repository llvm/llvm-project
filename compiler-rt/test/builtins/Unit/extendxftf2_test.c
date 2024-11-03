// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_extendxftf2

#include "int_lib.h"
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 64 && defined(__x86_64__) &&                          \
    (defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__))

#include "fp_test.h"

COMPILER_RT_ABI __float128 __extendxftf2(long double a);

int test__extendxftf2(long double a, uint64_t expectedHi, uint64_t expectedLo) {
  __float128 x = __extendxftf2(a);
  int ret = compareResultF128(x, expectedHi, expectedLo);

  if (ret) {
    printf("error in __extendxftf2(%.20Lf) = %.20Lf, "
           "expected %.20Lf\n",
           a, x, fromRep128(expectedHi, expectedLo));
  }
  return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main() {
#if __LDBL_MANT_DIG__ == 64 && defined(__x86_64__) &&                          \
    (defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__))
  // qNaN
  if (test__extendxftf2(makeQNaN80(), UINT64_C(0x7fff800000000000),
                        UINT64_C(0x0)))
    return 1;
  // NaN
  if (test__extendxftf2(makeNaN80(UINT64_C(0x3fffffffffffffff)),
                        UINT64_C(0x7fff7fffffffffff),
                        UINT64_C(0xfffe000000000000)))
    return 1;
  // inf
  if (test__extendxftf2(makeInf80(), UINT64_C(0x7fff000000000000),
                        UINT64_C(0x0)))
    return 1;
  // zero
  if (test__extendxftf2(0.0, UINT64_C(0x0), UINT64_C(0x0)))
    return 1;
  if (test__extendxftf2(0x1.23456789abcdefp+5, UINT64_C(0x400423456789abcd),
                        UINT64_C(0xf000000000000000)))
    return 1;
  if (test__extendxftf2(0x1.edcba987654321fp-9, UINT64_C(0x3ff6edcba9876543),
                        UINT64_C(0x2000000000000000)))
    return 1;
  if (test__extendxftf2(0x1.23456789abcdefp+45, UINT64_C(0x402c23456789abcd),
                        UINT64_C(0xf000000000000000)))
    return 1;
  if (test__extendxftf2(0x1.edcba987654321fp-45, UINT64_C(0x3fd2edcba9876543),
                        UINT64_C(0x2000000000000000)))
    return 1;
  // denormal number
  if (test__extendxftf2(1e-4932L, UINT64_C(0x00004c248f91e526),
                        UINT64_C(0xafe0000000000000)))
    return 1;
  // denormal number
  if (test__extendxftf2(2e-4932L, UINT64_C(0x000098491f23ca4d),
                        UINT64_C(0x5fc0000000000000)))
    return 1;
#else
  printf("skipped\n");

#endif
  return 0;
}
