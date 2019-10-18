// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_truncdfsf2

//===--------------- truncdfsf2_test.c - Test __truncdfsf2 ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __truncdfsf2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#include "fp_test.h"

float __truncdfsf2(double a);

int test__truncdfsf2(double a)
{
    float actual = __truncdfsf2(a);
    float expected = a;

    if (actual != expected) {
        printf("error in test__truncdfsf2(%lf) = %f, "
               "expected %f\n", a, actual, expected);
        return 1;
    }
    return 0;
}

int main()
{
    if (test__truncdfsf2(340282366920938463463374607431768211456.0))
        return 1;
    return 0;
}
