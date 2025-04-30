/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

/* ------------------------------- complex functions:  */

FLTFUNC_C(__mth_i_cabs)
{
  CMPLXARGS_C;
  CRETURN_F(hypotf(real, imag));
}
