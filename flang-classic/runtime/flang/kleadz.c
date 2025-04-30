/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdint.h>

int64_t
__mth_i_kleadz(int64_t i)
{
  uint64_t ui; /* unsigned representation of 'i' */
  int nz;          /* number of leading zero bits in 'i' */
  int k;

  ui = i;
  nz = 64;
  k = nz >> 1;
  while (k) {
    if (ui >> k) {
      ui >>= k;
      nz -= k;
    }
    k >>= 1;
  }
  if (ui)
    --nz;
  return nz;
}
