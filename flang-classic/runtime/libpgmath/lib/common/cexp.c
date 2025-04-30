/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

CMPLXFUNC_C(__mth_i_cexp)
{
  CMPLXARGS_C;
  float x, y, z;
  x = expf(real);
  __mth_sincos(imag, &z, &y);
  y *= x;
  z *= x;
  CRETURN_F_F(y, z); /* should leave y & z in appropriate
                  * registers */
}
