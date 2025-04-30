/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* Intrinsic function which take quad precision arguments. */

#include "mthdecls.h"

union {
  float128_t val;
  unsigned int num[4];
} tmp;


float128_t
__mth_i_qabs(float128_t arg)
{
  tmp.val = arg;
  /* little endian */
  if (is_little_endian()) {
    tmp.num[3] &= LITTLE_SIGN_MASK;
  } else {
    tmp.num[0] &= BIG_SIGN_MASK;
  }

  return tmp.val;
}
