// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_trunctfxf2

#include "int_lib.h"
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 64 && defined(__x86_64__) &&                          \
    (defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__))

#include "fp_test.h"

COMPILER_RT_ABI long double __trunctfxf2(tf_float a);

int test__trunctfxf2(tf_float a, uint64_t expectedHi, uint64_t expectedLo) {
  long double x = __trunctfxf2(a);
  int ret = compareResultF80(x, expectedHi, expectedLo);
  ;
  if (ret) {
    printf("error in __trunctfxf2(%.20Lf) = %.20Lf, "
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
  if (test__trunctfxf2(makeQNaN128(), UINT64_C(0x7FFF),
                       UINT64_C(0xC000000000000000)))
    return 1;
  // NaN
  if (test__trunctfxf2(makeNaN128(UINT64_C(0x810000000000)), UINT64_C(0x7FFF),
                       UINT64_C(0xC080000000000000)))
    return 1;
  // inf
  if (test__trunctfxf2(makeInf128(), UINT64_C(0x7FFF),
                       UINT64_C(0x8000000000000000)))
    return 1;
  // zero
  if (test__trunctfxf2(0.0Q, UINT64_C(0x0), UINT64_C(0x0)))
    return 1;
  if (test__trunctfxf2(0x1.af23456789bbaaab347645365cdep+5L, UINT64_C(0x4004),
                       UINT64_C(0xd791a2b3c4ddd556)))
    return 1;
  if (test__trunctfxf2(0x1.dedafcff354b6ae9758763545432p-9L, UINT64_C(0x3ff6),
                       UINT64_C(0xef6d7e7f9aa5b575)))
    return 1;
  if (test__trunctfxf2(0x1.2f34dd5f437e849b4baab754cdefp+4534L,
                       UINT64_C(0x51b5), UINT64_C(0x979a6eafa1bf424e)))
    return 1;
  if (test__trunctfxf2(0x1.edcbff8ad76ab5bf46463233214fp-435L, UINT64_C(0x3e4c),
                       UINT64_C(0xf6e5ffc56bb55ae0)))
    return 1;

  // Test rounding near halfway.
  tf_float halfwayPlus =
      fromRep128(UINT64_C(0x7ffa000000000000),
                 ((UINT64_C(1) << (112 - 63 - 1)) + UINT64_C(1)));
  if (test__trunctfxf2(halfwayPlus, UINT64_C(0x7ffa),
                       UINT64_C(0x8000000000000001)))
    return 1;
  tf_float halfwayExactOdd = fromRep128(
      UINT64_C(0x7ffa000000000000),
      ((UINT64_C(1) << (112 - 63)) + (UINT64_C(1) << (112 - 63 - 1))));
  if (test__trunctfxf2(halfwayExactOdd, UINT64_C(0x7ffa),
                       UINT64_C(0x8000000000000002)))
    return 1;
  tf_float halfwayExactEven =
      fromRep128(UINT64_C(0x7ffa000000000000), (UINT64_C(1) << (112 - 63 - 1)));
  if (test__trunctfxf2(halfwayExactEven, UINT64_C(0x7ffa),
                       UINT64_C(0x8000000000000000)))
    return 1;
  tf_float halfwayRoundingWillChangeExponent =
      fromRep128(UINT64_C(0x7ffaffffffffffff), UINT64_C(0xffff000000000001));
  if (test__trunctfxf2(halfwayRoundingWillChangeExponent, UINT64_C(0x7ffb),
                       UINT64_C(0x8000000000000000)))
    return 1;

  // denormal number
  if (test__trunctfxf2(1e-4932Q, UINT64_C(0), UINT64_C(0x261247c8f29357f0)))
    return 1;
  // denormal number
  if (test__trunctfxf2(2e-4932Q, UINT64_C(0), UINT64_C(0x4c248f91e526afe0)))
    return 1;

#else
  printf("skipped\n");

#endif
  return 0;
}
