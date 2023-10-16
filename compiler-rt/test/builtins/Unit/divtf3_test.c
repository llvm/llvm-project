// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_divtf3

#include "int_lib.h"
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

// Returns: a / b
COMPILER_RT_ABI long double __divtf3(long double a, long double b);

int test__divtf3(long double a, long double b,
                 uint64_t expectedHi, uint64_t expectedLo)
{
    long double x = __divtf3(a, b);
    int ret = compareResultF128(x, expectedHi, expectedLo);

    if (ret){
        printf("error in test__divtf3(%.20Le, %.20Le) = %.20Le, "
               "expected %.20Le\n", a, b, x,
               fromRep128(expectedHi, expectedLo));
    }
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    // Returned NaNs are assumed to be qNaN by default

    // qNaN / any = qNaN
    if (test__divtf3(makeQNaN128(),
                     0x1.23456789abcdefp+5L,
                     UINT64_C(0x7fff800000000000),
                     UINT64_C(0x0)))
        return 1;
    // NaN / any = NaN
    if (test__divtf3(makeNaN128(UINT64_C(0x30000000)),
                     0x1.23456789abcdefp+5L,
                     UINT64_C(0x7fff800000000000),
                     UINT64_C(0x0)))
        return 1;
    // any / qNaN = qNaN
    if (test__divtf3(0x1.23456789abcdefp+5L,
                     makeQNaN128(),
                     UINT64_C(0x7fff800000000000),
                     UINT64_C(0x0)))
        return 1;
    // any / NaN = NaN
    if (test__divtf3(0x1.23456789abcdefp+5L,
                     makeNaN128(UINT64_C(0x30000000)),
                     UINT64_C(0x7fff800000000000),
                     UINT64_C(0x0)))
        return 1;

    // +Inf / positive = +Inf
    if (test__divtf3(makeInf128(), 3.L,
                     UINT64_C(0x7fff000000000000),
                     UINT64_C(0x0)))
        return 1;
    // +Inf / negative = -Inf
    if (test__divtf3(makeInf128(), -3.L,
                     UINT64_C(0xffff000000000000),
                     UINT64_C(0x0)))
        return 1;
    // -Inf / positive = -Inf
    if (test__divtf3(makeNegativeInf128(), 3.L,
                     UINT64_C(0xffff000000000000),
                     UINT64_C(0x0)))
        return 1;
    // -Inf / negative = +Inf
    if (test__divtf3(makeNegativeInf128(), -3.L,
                     UINT64_C(0x7fff000000000000),
                     UINT64_C(0x0)))
        return 1;

    // Inf / Inf = NaN
    if (test__divtf3(makeInf128(), makeInf128(),
                     UINT64_C(0x7fff800000000000),
                     UINT64_C(0x0)))
        return 1;
    // 0.0 / 0.0 = NaN
    if (test__divtf3(+0x0.0p+0L, +0x0.0p+0L,
                     UINT64_C(0x7fff800000000000),
                     UINT64_C(0x0)))
        return 1;
    // +0.0 / +Inf = +0.0
    if (test__divtf3(+0x0.0p+0L, makeInf128(),
                     UINT64_C(0x0),
                     UINT64_C(0x0)))
        return 1;
    // +Inf / +0.0 = +Inf
    if (test__divtf3(makeInf128(), +0x0.0p+0L,
                     UINT64_C(0x7fff000000000000),
                     UINT64_C(0x0)))
        return 1;

    // positive / +0.0 = +Inf
    if (test__divtf3(+1.0L, +0x0.0p+0L,
                     UINT64_C(0x7fff000000000000),
                     UINT64_C(0x0)))
        return 1;
    // positive / -0.0 = -Inf
    if (test__divtf3(+1.0L, -0x0.0p+0L,
                     UINT64_C(0xffff000000000000),
                     UINT64_C(0x0)))
        return 1;
    // negative / +0.0 = -Inf
    if (test__divtf3(-1.0L, +0x0.0p+0L,
                     UINT64_C(0xffff000000000000),
                     UINT64_C(0x0)))
        return 1;
    // negative / -0.0 = +Inf
    if (test__divtf3(-1.0L, -0x0.0p+0L,
                     UINT64_C(0x7fff000000000000),
                     UINT64_C(0x0)))
        return 1;

    // 1/3
    if (test__divtf3(1.L, 3.L,
                     UINT64_C(0x3ffd555555555555),
                     UINT64_C(0x5555555555555555)))
        return 1;
    // smallest normal result
    if (test__divtf3(0x1.0p-16381L, 2.L,
                     UINT64_C(0x0001000000000000),
                     UINT64_C(0x0)))
        return 1;

    // divisor is exactly 1.0
    if (test__divtf3(0x1.0p+0L,
                     0x1.0p+0L,
                     UINT64_C(0x3fff000000000000),
                     UINT64_C(0x0)))
        return 1;
    // divisor is truncated to exactly 1.0 in UQ1.63
    if (test__divtf3(0x1.0p+0L,
                     0x1.0000000000000001p+0L,
                     UINT64_C(0x3ffeffffffffffff),
                     UINT64_C(0xfffe000000000000)))
        return 1;

    // smallest normal value divided by 2.0
    if (test__divtf3(0x1.0p-16382L, 2.L, UINT64_C(0x0000800000000000), UINT64_C(0x0)))
      return 1;
    // smallest subnormal result
    if (test__divtf3(0x1.0p-16382L, 0x1p+112L, UINT64_C(0x0), UINT64_C(0x1)))
      return 1;

    // any / any
    if (test__divtf3(0x1.a23b45362464523375893ab4cdefp+5L,
                     0x1.eedcbaba3a94546558237654321fp-1L,
                     UINT64_C(0x4004b0b72924d407),
                     UINT64_C(0x0717e84356c6eba2)))
        return 1;
    if (test__divtf3(0x1.a2b34c56d745382f9abf2c3dfeffp-50L,
                     0x1.ed2c3ba15935332532287654321fp-9L,
                     UINT64_C(0x3fd5b2af3f828c9b),
                     UINT64_C(0x40e51f64cde8b1f2)))
        return 15;
    if (test__divtf3(0x1.2345f6aaaa786555f42432abcdefp+456L,
                     0x1.edacbba9874f765463544dd3621fp+6400L,
                     UINT64_C(0x28c62e15dc464466),
                     UINT64_C(0xb5a07586348557ac)))
        return 1;
    if (test__divtf3(0x1.2d3456f789ba6322bc665544edefp-234L,
                     0x1.eddcdba39f3c8b7a36564354321fp-4455L,
                     UINT64_C(0x507b38442b539266),
                     UINT64_C(0x22ce0f1d024e1252)))
        return 1;
    if (test__divtf3(0x1.2345f6b77b7a8953365433abcdefp+234L,
                     0x1.edcba987d6bb3aa467754354321fp-4055L,
                     UINT64_C(0x50bf2e02f0798d36),
                     UINT64_C(0x5e6fcb6b60044078)))
        return 1;
    if (test__divtf3(6.72420628622418701252535563464350521E-4932L,
                     2.L,
                     UINT64_C(0x0001000000000000),
                     UINT64_C(0)))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}
