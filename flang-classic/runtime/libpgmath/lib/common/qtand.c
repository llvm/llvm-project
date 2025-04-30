/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* Intrinsic function which take quad precision arguments. */

#include <stdint.h>
#include "mthdecls.h"
#define MYPI 180.0
#define HFPI 90.0

float128_t
__mth_i_qtand(float128_t d)
{
  union {
    float128_t q;
    ui64arr2_t i;
  } u, sp;
  if ((fmodl(fabsl(d), MYPI) - HFPI) == 0.0) {
    sp.i[0] = is_little_endian() ? 0x0 : 0x7fff000000000000;
    sp.i[1] = is_little_endian() ? 0x7fff000000000000 : 0x0;
    return sp.q;
  }
  /* if the host is little endian */
  if (is_little_endian()) {
    /* value of pi/180 = 0.01745329251994329576923690768489 */
    u.i[0] = 0x915C1D8BECDD290B;
    u.i[1] = 0x3FF91DF46A2529D3;
  } else { /* big endian */
    u.i[0] = 0xD329256AF41DF93F;
    u.i[1] = 0x0B29DDEC8B1D5C91;
  }
  return tanl(d * u.q);
}
