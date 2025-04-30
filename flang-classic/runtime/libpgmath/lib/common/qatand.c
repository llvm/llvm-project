/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* Intrinsic function which take quad precision arguments. */

#include <stdint.h>
#include "mthdecls.h"
float128_t
__mth_i_qatand(float128_t d)
{
  union {
    float128_t q;
    ui64arr2_t i;
  } u;

  /* if the host is little endian */
  if (is_little_endian()) {
    /* value of 180/pi = 57.29577951308232087679815481410517 */
    u.i[0] = 0x7B86152EA6FE81A5;
    u.i[1] = 0x4004CA5DC1A63C1F;
  } else { /* big endian */
    u.i[0] = 0x1F3CA6C15DCA0440;
    u.i[1] = 0xA581FEA62E15867B;
  }
  return atanl(d) * u.q;
}
