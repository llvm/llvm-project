/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Part of the f2008 complex cosh intrinsic test
 */

#include <stdio.h>
#include <complex.h>

extern float complex ccoshf(float complex);
extern double complex ccosh(double complex);

void
get_expected_cf(complex float src1[], complex float expct[], int n)
{
    int i;

    for(i= 0; i <n; i++ ) {
        expct[i] = ccosh(src1[i]);
        /*printf("%d) ynf(%e) = %e\n", i, src1[i], expct[i]);*/
    }
}

void
get_expected_cd(complex double src1[], complex double expct[], int n)
{
    int i;

    for(i= 0; i <n; i++ ) {
        expct[i] = ccosh(src1[i]);
        /*printf("%d) yn(%e) = %e\n", i, src1[i], expct[i]);*/
    }
}
