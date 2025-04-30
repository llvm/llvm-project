/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Part of the f2008 bessel_yn tranformational intrinsic test
 */

#include <stdio.h>
#include <math.h>

void
get_expected_f(float src1, float expct[], int n1, int n2, int n)
{
    int i;
    int order;

        for(order = n1; order <= n2; order++ ) {
#if  defined( _WIN64) || defined(_WIN32) || defined(osx86)
            switch(order) {
            case 0:
                expct[order-n1] = (float)y0(src1);
                break;
            case 1:
                expct[order-n1] = (float)y1(src1);
                break;
            default:
                expct[order-n1] = (float)yn(order, src1);
                break;
            }
#else
            switch(order) {
            case 0:
                expct[order-n1] = (float)y0f(src1);
                break;
            case 1:
                expct[order-n1] = (float)y1f(src1);
                break;
            default:
                expct[order-n1] = (float)ynf(order, src1);
                break;
            }
#endif
            /*printf("%d) ynf(%e) = %e\n", i, src1, expct[i+order-n1]);*/
        }
}

void
get_expected_d(double src1, double expct[], int n1, int n2, int n)
{
    int i;
    int order;

        for(order = n1; order <= n2; order++ ) {
            switch(order) {
            case 0:
                expct[order-n1] = y0(src1);
                break;
            case 1:
                expct[order-n1] = y1(src1);
                break;
            default:
                expct[order-n1] = yn(order, src1);
                break;
            }
            /*printf("%d) yn(%e) = %e\n", i, src1, expct[i+order-n1]);*/
        }
}
