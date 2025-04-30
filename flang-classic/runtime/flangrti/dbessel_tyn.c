/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* bessel_tyn.c implements float F2008 bessel_yn transformational intrinsic */

double __mth_i_dbessel_y0(double arg);
double __mth_i_dbessel_y1(double arg);
double __mth_i_dbessel_yn(int n, double arg);

void
f90_dbessel_yn(double *rslts, int *n1, int *n2, double *x)
{
  int i;
  double *rslt_p;

  for (i = *n1, rslt_p = rslts; i <= *n2; i++, rslt_p++) {
    switch (i) {
    case 0:
      *rslt_p = __mth_i_dbessel_y0(*x);
      break;
    case 1:
      *rslt_p = __mth_i_dbessel_y1(*x);
      break;
    default:
      *rslt_p = __mth_i_dbessel_yn(i, *x);
      break;
    }
  }
}

