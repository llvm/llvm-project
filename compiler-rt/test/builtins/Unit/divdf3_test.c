// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_divdf3

#include "int_lib.h"
#include <stdio.h>

#include "fp_test.h"

// Returns: a / b
COMPILER_RT_ABI double __divdf3(double a, double b);

int test__divdf3(double a, double b, uint64_t expected)
{
    double x = __divdf3(a, b);
    int ret = compareResultD(x, expected);

    if (ret){
        printf("error in test__divdf3(%.20e, %.20e) = %.20e, "
               "expected %.20e\n", a, b, x,
               fromRep64(expected));
    }
    return ret;
}

int main()
{
    // Returned NaNs are assumed to be qNaN by default

    // qNaN / any = qNaN
    if (test__divdf3(makeQNaN64(), 3., UINT64_C(0x7ff8000000000000)))
      return 1;
    // NaN / any = NaN
    if (test__divdf3(makeNaN64(UINT64_C(0x123)), 3., UINT64_C(0x7ff8000000000000)))
      return 1;
    // any / qNaN = qNaN
    if (test__divdf3(3., makeQNaN64(), UINT64_C(0x7ff8000000000000)))
      return 1;
    // any / NaN = NaN
    if (test__divdf3(3., makeNaN64(UINT64_C(0x123)), UINT64_C(0x7ff8000000000000)))
      return 1;

    // +Inf / positive = +Inf
    if (test__divdf3(makeInf64(), 3., UINT64_C(0x7ff0000000000000)))
      return 1;
    // +Inf / negative = -Inf
    if (test__divdf3(makeInf64(), -3., UINT64_C(0xfff0000000000000)))
      return 1;
    // -Inf / positive = -Inf
    if (test__divdf3(makeNegativeInf64(), 3., UINT64_C(0xfff0000000000000)))
      return 1;
    // -Inf / negative = +Inf
    if (test__divdf3(makeNegativeInf64(), -3., UINT64_C(0x7ff0000000000000)))
      return 1;

    // Inf / Inf = NaN
    if (test__divdf3(makeInf64(), makeInf64(), UINT64_C(0x7ff8000000000000)))
      return 1;
    // 0.0 / 0.0 = NaN
    if (test__divdf3(+0x0.0p+0, +0x0.0p+0, UINT64_C(0x7ff8000000000000)))
      return 1;
    // +0.0 / +Inf = +0.0
    if (test__divdf3(+0x0.0p+0, makeInf64(), UINT64_C(0x0)))
      return 1;
    // +Inf / +0.0 = +Inf
    if (test__divdf3(makeInf64(), +0x0.0p+0, UINT64_C(0x7ff0000000000000)))
      return 1;

    // positive / +0.0 = +Inf
    if (test__divdf3(+1.0, +0x0.0p+0, UINT64_C(0x7ff0000000000000)))
      return 1;
    // positive / -0.0 = -Inf
    if (test__divdf3(+1.0, -0x0.0p+0, UINT64_C(0xfff0000000000000)))
      return 1;
    // negative / +0.0 = -Inf
    if (test__divdf3(-1.0, +0x0.0p+0, UINT64_C(0xfff0000000000000)))
      return 1;
    // negative / -0.0 = +Inf
    if (test__divdf3(-1.0, -0x0.0p+0, UINT64_C(0x7ff0000000000000)))
      return 1;

    // 1/3
    if (test__divdf3(1., 3., UINT64_C(0x3fd5555555555555)))
      return 1;
    // smallest normal result
    if (test__divdf3(0x1.0p-1021, 2., UINT64_C(0x10000000000000)))
      return 1;

    // divisor is exactly 1.0
    if (test__divdf3(0x1.0p+0, 0x1.0p+0, UINT64_C(0x3ff0000000000000)))
      return 1;
    // divisor is truncated to exactly 1.0 in UQ1.31
    if (test__divdf3(0x1.0p+0, 0x1.00000001p+0, UINT64_C(0x3fefffffffe00000)))
      return 1;

    return 0;
}
