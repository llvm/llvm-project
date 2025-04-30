/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Part of the f2008 complex asin intrinsic test
 */

#include <stdio.h>
#include <complex.h>

extern float complex casinf(float complex);
extern double complex casin(double complex);

#undef USELIBMF
#if !defined(_WIN64) && defined(__PGIC__) && defined(__PGIC_MINOR__)
#if (__PGIC__ > 15)
#define USELIBMF
#elif (__PGIC__ == 15) && (__PGIC_MINOR_ > 4)
#define USELIBMF
#endif
#endif

void
get_expected_cf(complex float src1[], complex float expct[], int n)
{
    int i;

    for(i= 0; i <n; i++ ) {
#ifdef USELIBMF
        expct[i] = casinf(src1[i]);
#else
        expct[i] = casin(src1[i]);
#endif
        /*printf("%d) ynf(%e) = %e\n", i, src1[i], expct[i]);*/
    }
}

void
get_expected_cd(complex double src1[], complex double expct[], int n)
{
    int i;

    for(i= 0; i <n; i++ ) {
        expct[i] = casin(src1[i]);
        /*printf("%d) yn(%e) = %e\n", i, src1[i], expct[i]);*/
    }
}
