/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

/*  ----------------------------- long double complex functions:  */

QUADFUNC_C(__mth_i_cqabs)
{
  LZMPLXARGS_LZ;
  LZRETURN_Q(hypotl(real, imag));
}
