/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

/* For X86-64 architectures, cdexp is defined in fastmath.s */

#if (! defined (TARGET_X8664) && ! defined(LINUX8664))
ZMPLXFUNC_Z(__mth_i_cdexp)
{
  ZMPLXARGS_Z;
  double x, y, z;
  x = exp(real);
  __mth_dsincos(imag, &z, &y);
  y *= x;
  z *= x;
  ZRETURN_D_D(y, z); /* should leave y & z in appropriate
                  * registers */
}
#endif
