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
    // 1/3
    if (test__divsf3(1.f, 3.f, 0x3EAAAAABU))
      return 1;
    // smallest normal result
    if (test__divsf3(2.3509887e-38, 2., 0x00800000U))
      return 1;

    return 0;
}
