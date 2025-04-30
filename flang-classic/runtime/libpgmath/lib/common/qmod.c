/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* Intrinsic function which take quad precision arguments. */

#include "mthdecls.h"

#ifndef WIN64
float128_t
__mth_i_qmod(float128_t f, float128_t g)
{
  return fmodl(f, g);
}
#endif
