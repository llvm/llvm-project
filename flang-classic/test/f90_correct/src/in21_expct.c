/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Part of the f2008 bessel_jn intrinsic test
 */

#include <stdio.h>
#include <math.h>

void
get_expected_f(float src1[], float expct[], int order, int n)
{
    int i;

    for(i= 0; i <n; i++ ) {
#if  defined( _WIN64) ||  defined(_WIN32) || defined(osx86)
        expct[i] = jn(order, src1[i]);
#else
        expct[i] = jnf(order, src1[i]);
#endif
        /*printf("%d) jnf(%e) = %e\n", i, src1[i], expct[i]);*/
    }
}

void
get_expected_d(double src1[], double expct[], int order, int n)
{
    int i;

    for(i= 0; i <n; i++ ) {
        expct[i] = jn(order, src1[i]);
        /*printf("%d) jn(%e) = %e\n", i, src1[i], expct[i]);*/
    }
}
