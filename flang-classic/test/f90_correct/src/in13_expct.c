
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Part of the f2008 hypot intrinsic test
 */

#include <stdio.h>
#include <math.h>

void
get_expected_f(float src1[], float src2[], float expct[], int n)
{
    int i;

    for(i= 0; i <n; i++ ) {
#ifdef  WIN64
        expct[i] = hypot(src1[i],src2[i]);
#else
        expct[i] = hypotf(src1[i], src2[i]);
#endif
        /*printf("%d) hypotf(%e, %e) = %e\n",i, src1[i],src2[i], expct[i]);*/
    }
}

void
get_expected_d(double src1[], double src2[], double expct[], int n)
{
    int i;

    for(i= 0; i <n; i++ ) {
        expct[i] = hypot(src1[i],src2[i]);
        /*printf("%d) hypot(%e, %e) = %e\n",i, src1[i], src2[i]. expct[i]);*/
    }
}
