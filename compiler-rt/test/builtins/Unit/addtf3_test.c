// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_addtf3

#include <fenv.h>
#include <stdio.h>

// The testcase currently assumes IEEE TF format, once that has been
// fixed the defined(CRT_HAS_IEEE_TF) guard can be removed to enable it for
// IBM 128 floats as well.
#if defined(CRT_HAS_IEEE_TF)

#  include "fp_test.h"
#  include "int_lib.h"

// Returns: a + b
COMPILER_RT_ABI tf_float __addtf3(tf_float a, tf_float b);

int test__addtf3(tf_float a, tf_float b, uint64_t expectedHi,
                 uint64_t expectedLo) {
  tf_float x = __addtf3(a, b);
  int ret = compareResultF128(x, expectedHi, expectedLo);

  if (ret) {
    printf("error in test__addtf3(%.20Lf, %.20Lf) = %.20Lf, "
           "expected %.20Lf\n",
           a, b, x, fromRep128(expectedHi, expectedLo));
  }

  return ret;
}

char assumption_1[sizeof(tf_float) * CHAR_BIT == 128] = {0};

#endif

int main() {
#if defined(CRT_HAS_IEEE_TF)
  // qNaN + any = qNaN
  if (test__addtf3(makeQNaN128(), 0x1.23456789abcdefp+5L,
                   UINT64_C(0x7fff800000000000), UINT64_C(0x0)))
    return 1;
  // NaN + any = NaN
  if (test__addtf3(makeNaN128(UINT64_C(0x800030000000)),
                   TF_C(0x1.23456789abcdefp+5), UINT64_C(0x7fff800000000000),
                   UINT64_C(0x0)))
    return 1;
  // inf + inf = inf
  if (test__addtf3(makeInf128(), makeInf128(), UINT64_C(0x7fff000000000000),
                   UINT64_C(0x0)))
    return 1;
  // inf + any = inf
  if (test__addtf3(makeInf128(), TF_C(0x1.2335653452436234723489432abcdefp+5),
                   UINT64_C(0x7fff000000000000), UINT64_C(0x0)))
    return 1;
  // any + any
  if (test__addtf3(TF_C(0x1.23456734245345543849abcdefp+5),
                   TF_C(0x1.edcba52449872455634654321fp-1),
                   UINT64_C(0x40042afc95c8b579), UINT64_C(0x61e58dd6c51eb77c)))
    return 1;

#  if (defined(__arm__) || defined(__aarch64__)) && defined(__ARM_FP) ||       \
      defined(i386) || defined(__x86_64__) ||                                  \
      (defined(__loongarch__) && __loongarch_frlen != 0)
  // Rounding mode tests on supported architectures
  const tf_float m = 1234.0L, n = 0.01L;

  fesetround(FE_UPWARD);
  if (test__addtf3(m, n, UINT64_C(0x40093480a3d70a3d),
                   UINT64_C(0x70a3d70a3d70a3d8)))
    return 1;

  fesetround(FE_DOWNWARD);
  if (test__addtf3(m, n, UINT64_C(0x40093480a3d70a3d),
                   UINT64_C(0x70a3d70a3d70a3d7)))
    return 1;

  fesetround(FE_TOWARDZERO);
  if (test__addtf3(m, n, UINT64_C(0x40093480a3d70a3d),
                   UINT64_C(0x70a3d70a3d70a3d7)))
    return 1;

  fesetround(FE_TONEAREST);
  if (test__addtf3(m, n, UINT64_C(0x40093480a3d70a3d),
                   UINT64_C(0x70a3d70a3d70a3d7)))
    return 1;
#  endif

#else
  printf("skipped\n");

#endif
  return 0;
}
