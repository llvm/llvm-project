/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Part of the f2008 asinh intrinsic test
 */

#include <stdio.h>
#include <math.h>

void
get_expected_f(float src1[], float expct[], int n)
{
    int i;

    for(i= 0; i <n; i++ ) {
#ifdef  _WIN64
        expct[i] = asinh(src1[i]);
#else
        expct[i] = asinhf(src1[i]);
#endif
        /*printf("%d) asinhf(%e) = %e\n", i, src1[i], expct[i]);*/
    }
}

void
get_expected_d(double src1[], double expct[], int n)
{
    int i;

    for(i= 0; i <n; i++ ) {
        expct[i] = asinh(src1[i]);
        /*printf("%d) asinh(%e) = %e\n", i, src1[i], expct[i]);*/
    }
}
