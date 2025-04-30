/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* bessel_tjn.c implements float F2008 bessel_jn transformational intrinsic */

float __mth_i_bessel_j0(float arg);
float __mth_i_bessel_j1(float arg);
float __mth_i_bessel_jn(int n, float arg);

void
f90_bessel_jn(float *rslts, int *n1, int *n2, float *x)
{
  int i;
  float *rslt_p;

  for (i = *n1, rslt_p = rslts; i <= *n2; i++, rslt_p++) {
    switch (i) {
    case 0:
      *rslt_p = __mth_i_bessel_j0(*x);
      break;
    case 1:
      *rslt_p = __mth_i_bessel_j1(*x);
      break;
    default:
      *rslt_p = __mth_i_bessel_jn(i, *x);
      break;
    }
  }
}

