// RUN: %clang_builtins %s %librt -o %t && %run %t
//===--------------- trunctfsf2_test.c - Test __trunctfsf2 ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __trunctfsf2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

COMPILER_RT_ABI float __trunctfsf2(long double a);

int test__trunctfsf2(long double a, uint32_t expected)
{
    float x = __trunctfsf2(a);
    int ret = compareResultF(x, expected);

    if (ret){
        printf("error in test__trunctfsf2(%.20Lf) = %f, "
               "expected %f\n", a, x, fromRep32(expected));
    }
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    // qNaN
    if (test__trunctfsf2(makeQNaN128(),
                         UINT32_C(0x7fc00000)))
        return 1;
    // NaN
    if (test__trunctfsf2(makeNaN128(UINT64_C(0x810000000000)),
                         UINT32_C(0x7fc08000)))
        return 1;
    // inf
    if (test__trunctfsf2(makeInf128(),
                         UINT32_C(0x7f800000)))
        return 1;
    // zero
    if (test__trunctfsf2(0.0L, UINT32_C(0x0)))
        return 1;

    if (test__trunctfsf2(0x1.23a2abb4a2ddee355f36789abcdep+5L,
                         UINT32_C(0x4211d156)))
        return 1;
    if (test__trunctfsf2(0x1.e3d3c45bd3abfd98b76a54cc321fp-9L,
                         UINT32_C(0x3b71e9e2)))
        return 1;
    if (test__trunctfsf2(0x1.234eebb5faa678f4488693abcdefp+4534L,
                         UINT32_C(0x7f800000)))
        return 1;
    if (test__trunctfsf2(0x1.edcba9bb8c76a5a43dd21f334634p-435L,
                         UINT32_C(0x0)))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}
