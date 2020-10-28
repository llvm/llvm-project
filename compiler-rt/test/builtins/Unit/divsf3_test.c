// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_divsf3

#include "int_lib.h"
#include <stdio.h>

#include "fp_test.h"

// Returns: a / b
COMPILER_RT_ABI float __divsf3(float a, float b);

int test__divsf3(float a, float b, uint32_t expected)
{
    float x = __divsf3(a, b);
    int ret = compareResultF(x, expected);

    if (ret){
        printf("error in test__divsf3(%.20e, %.20e) = %.20e, "
               "expected %.20e\n", a, b, x,
               fromRep32(expected));
    }
    return ret;
}

int main()
{
    // Returned NaNs are assumed to be qNaN by default

    // qNaN / any = qNaN
    if (test__divsf3(makeQNaN32(), 3.F, UINT32_C(0x7fc00000)))
      return 1;
    // NaN / any = NaN
    if (test__divsf3(makeNaN32(UINT32_C(0x123)), 3.F, UINT32_C(0x7fc00000)))
      return 1;
    // any / qNaN = qNaN
    if (test__divsf3(3.F, makeQNaN32(), UINT32_C(0x7fc00000)))
      return 1;
    // any / NaN = NaN
    if (test__divsf3(3.F, makeNaN32(UINT32_C(0x123)), UINT32_C(0x7fc00000)))
      return 1;

    // +Inf / positive = +Inf
    if (test__divsf3(makeInf32(), 3.F, UINT32_C(0x7f800000)))
      return 1;
    // +Inf / negative = -Inf
    if (test__divsf3(makeInf32(), -3.F, UINT32_C(0xff800000)))
      return 1;
    // -Inf / positive = -Inf
    if (test__divsf3(makeNegativeInf32(), 3.F, UINT32_C(0xff800000)))
      return 1;
    // -Inf / negative = +Inf
    if (test__divsf3(makeNegativeInf32(), -3.F, UINT32_C(0x7f800000)))
      return 1;

    // Inf / Inf = NaN
    if (test__divsf3(makeInf32(), makeInf32(), UINT32_C(0x7fc00000)))
      return 1;
    // 0.0 / 0.0 = NaN
    if (test__divsf3(+0x0.0p+0F, +0x0.0p+0F, UINT32_C(0x7fc00000)))
      return 1;
    // +0.0 / +Inf = +0.0
    if (test__divsf3(+0x0.0p+0F, makeInf32(), UINT32_C(0x0)))
      return 1;
    // +Inf / +0.0 = +Inf
    if (test__divsf3(makeInf32(), +0x0.0p+0F, UINT32_C(0x7f800000)))
      return 1;

    // positive / +0.0 = +Inf
    if (test__divsf3(+1.F, +0x0.0p+0F, UINT32_C(0x7f800000)))
      return 1;
    // positive / -0.0 = -Inf
    if (test__divsf3(+1.F, -0x0.0p+0F, UINT32_C(0xff800000)))
      return 1;
    // negative / +0.0 = -Inf
    if (test__divsf3(-1.F, +0x0.0p+0F, UINT32_C(0xff800000)))
      return 1;
    // negative / -0.0 = +Inf
    if (test__divsf3(-1.F, -0x0.0p+0F, UINT32_C(0x7f800000)))
      return 1;

    // 1/3
    if (test__divsf3(1.F, 3.F, UINT32_C(0x3eaaaaab)))
      return 1;
    // smallest normal result
    if (test__divsf3(0x1.0p-125F, 2.F, UINT32_C(0x00800000)))
      return 1;

    // divisor is exactly 1.0
    if (test__divsf3(0x1.0p+0F, 0x1.0p+0F, UINT32_C(0x3f800000)))
      return 1;
    // divisor is truncated to exactly 1.0 in UQ1.15
    if (test__divsf3(0x1.0p+0F, 0x1.0001p+0F, UINT32_C(0x3f7fff00)))
      return 1;

    return 0;
}
