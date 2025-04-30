/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Part of the f2008 log_gamma intrinsic test
 */

#include <stdio.h>
#include <math.h>

void
get_expected_f(float src[], float expct[], int n)
{
    int i;

    for(i= 0; i <n; i++ ) {
#ifdef  WIN64
        expct[i] = lgamma(src[i]);
#else
        expct[i] = lgammaf(src[i]);
#endif
        /*printf("%d) lgammaf(%e) = %e\n",i, src[i], expct[i]);*/
    }
}

void
get_expected_d(double src[], double expct[], int n)
{
    int i;

    for(i= 0; i <n; i++ ) {
        expct[i] = lgamma(src[i]);
        /*printf("%d) lgamma(%e) = %e\n",i, src[i], expct[i]);*/
    }
}
