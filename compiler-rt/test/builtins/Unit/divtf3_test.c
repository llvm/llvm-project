// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_divtf3

#include "int_lib.h"
#include <stdio.h>

// The testcase currently assumes IEEE TF format, once that has been
// fixed the defined(CRT_HAS_IEEE_TF) guard can be removed to enable it for
// IBM 128 floats as well.
#if defined(CRT_HAS_IEEE_TF)

#  include "fp_test.h"

// Returns: a / b
COMPILER_RT_ABI tf_float __divtf3(tf_float a, tf_float b);

int test__divtf3(tf_float a, tf_float b, uint64_t expectedHi,
                 uint64_t expectedLo) {
  tf_float x = __divtf3(a, b);
  int ret = compareResultF128(x, expectedHi, expectedLo);

  if (ret) {
    printf("error in test__divtf3(%.20Le, %.20Le) = %.20Le, "
           "expected %.20Le\n",
           a, b, x, fromRep128(expectedHi, expectedLo));
  }
  return ret;
}

char assumption_1[sizeof(tf_float) * CHAR_BIT == 128] = {0};

#endif

int main() {
#if defined(CRT_HAS_IEEE_TF)
  // Returned NaNs are assumed to be qNaN by default

  // qNaN / any = qNaN
  if (test__divtf3(makeQNaN128(), TF_C(0x1.23456789abcdefp+5),
                   UINT64_C(0x7fff800000000000), UINT64_C(0x0)))
    return 1;
  // NaN / any = NaN
  if (test__divtf3(makeNaN128(UINT64_C(0x30000000)),
                   TF_C(0x1.23456789abcdefp+5), UINT64_C(0x7fff800000000000),
                   UINT64_C(0x0)))
    return 1;
  // any / qNaN = qNaN
  if (test__divtf3(TF_C(0x1.23456789abcdefp+5), makeQNaN128(),
                   UINT64_C(0x7fff800000000000), UINT64_C(0x0)))
    return 1;
  // any / NaN = NaN
  if (test__divtf3(TF_C(0x1.23456789abcdefp+5),
                   makeNaN128(UINT64_C(0x30000000)),
                   UINT64_C(0x7fff800000000000), UINT64_C(0x0)))
    return 1;

  // +Inf / positive = +Inf
  if (test__divtf3(makeInf128(), TF_C(3.), UINT64_C(0x7fff000000000000),
                   UINT64_C(0x0)))
    return 1;
  // +Inf / negative = -Inf
  if (test__divtf3(makeInf128(), -TF_C(3.), UINT64_C(0xffff000000000000),
                   UINT64_C(0x0)))
    return 1;
  // -Inf / positive = -Inf
  if (test__divtf3(makeNegativeInf128(), TF_C(3.), UINT64_C(0xffff000000000000),
                   UINT64_C(0x0)))
    return 1;
  // -Inf / negative = +Inf
  if (test__divtf3(makeNegativeInf128(), -TF_C(3.),
                   UINT64_C(0x7fff000000000000), UINT64_C(0x0)))
    return 1;

  // Inf / Inf = NaN
  if (test__divtf3(makeInf128(), makeInf128(), UINT64_C(0x7fff800000000000),
                   UINT64_C(0x0)))
    return 1;
  // 0.0 / 0.0 = NaN
  if (test__divtf3(+TF_C(0x0.0p+0), +TF_C(0x0.0p+0),
                   UINT64_C(0x7fff800000000000), UINT64_C(0x0)))
    return 1;
  // +0.0 / +Inf = +0.0
  if (test__divtf3(+TF_C(0x0.0p+0), makeInf128(), UINT64_C(0x0), UINT64_C(0x0)))
    return 1;
  // +Inf / +0.0 = +Inf
  if (test__divtf3(makeInf128(), +TF_C(0x0.0p+0), UINT64_C(0x7fff000000000000),
                   UINT64_C(0x0)))
    return 1;

  // positive / +0.0 = +Inf
  if (test__divtf3(+TF_C(1.0), +TF_C(0x0.0p+0), UINT64_C(0x7fff000000000000),
                   UINT64_C(0x0)))
    return 1;
  // positive / -0.0 = -Inf
  if (test__divtf3(+1.0L, -TF_C(0x0.0p+0), UINT64_C(0xffff000000000000),
                   UINT64_C(0x0)))
    return 1;
  // negative / +0.0 = -Inf
  if (test__divtf3(-1.0L, +TF_C(0x0.0p+0), UINT64_C(0xffff000000000000),
                   UINT64_C(0x0)))
    return 1;
  // negative / -0.0 = +Inf
  if (test__divtf3(TF_C(-1.0), -TF_C(0x0.0p+0), UINT64_C(0x7fff000000000000),
                   UINT64_C(0x0)))
    return 1;

  // 1/3
  if (test__divtf3(TF_C(1.), TF_C(3.), UINT64_C(0x3ffd555555555555),
                   UINT64_C(0x5555555555555555)))
    return 1;
  // smallest normal result
  if (test__divtf3(TF_C(0x1.0p-16381), TF_C(2.), UINT64_C(0x0001000000000000),
                   UINT64_C(0x0)))
    return 1;

  // divisor is exactly 1.0
  if (test__divtf3(TF_C(0x1.0p+0), TF_C(0x1.0p+0), UINT64_C(0x3fff000000000000),
                   UINT64_C(0x0)))
    return 1;
  // divisor is truncated to exactly 1.0 in UQ1.63
  if (test__divtf3(TF_C(0x1.0p+0), TF_C(0x1.0000000000000001p+0),
                   UINT64_C(0x3ffeffffffffffff), UINT64_C(0xfffe000000000000)))
    return 1;

  // smallest normal value divided by 2.0
  if (test__divtf3(TF_C(0x1.0p-16382), 2.L, UINT64_C(0x0000800000000000),
                   UINT64_C(0x0)))
    return 1;
  // smallest subnormal result
  if (test__divtf3(TF_C(0x1.0p-16382), TF_C(0x1p+112), UINT64_C(0x0),
                   UINT64_C(0x1)))
    return 1;

  // any / any
  if (test__divtf3(TF_C(0x1.a23b45362464523375893ab4cdefp+5),
                   TF_C(0x1.eedcbaba3a94546558237654321fp-1),
                   UINT64_C(0x4004b0b72924d407), UINT64_C(0x0717e84356c6eba2)))
    return 1;
  if (test__divtf3(TF_C(0x1.a2b34c56d745382f9abf2c3dfeffp-50),
                   TF_C(0x1.ed2c3ba15935332532287654321fp-9),
                   UINT64_C(0x3fd5b2af3f828c9b), UINT64_C(0x40e51f64cde8b1f2)))
    return 15;
  if (test__divtf3(TF_C(0x1.2345f6aaaa786555f42432abcdefp+456),
                   TF_C(0x1.edacbba9874f765463544dd3621fp+6400),
                   UINT64_C(0x28c62e15dc464466), UINT64_C(0xb5a07586348557ac)))
    return 1;
  if (test__divtf3(TF_C(0x1.2d3456f789ba6322bc665544edefp-234),
                   TF_C(0x1.eddcdba39f3c8b7a36564354321fp-4455),
                   UINT64_C(0x507b38442b539266), UINT64_C(0x22ce0f1d024e1252)))
    return 1;
  if (test__divtf3(TF_C(0x1.2345f6b77b7a8953365433abcdefp+234),
                   TF_C(0x1.edcba987d6bb3aa467754354321fp-4055),
                   UINT64_C(0x50bf2e02f0798d36), UINT64_C(0x5e6fcb6b60044078)))
    return 1;
  if (test__divtf3(TF_C(6.72420628622418701252535563464350521E-4932), TF_C(2.),
                   UINT64_C(0x0001000000000000), UINT64_C(0)))
    return 1;

#else
  printf("skipped\n");

#endif
  return 0;
}
