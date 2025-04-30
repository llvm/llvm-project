/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

int
__mth_i_ileadz(int i)
{
  unsigned ui;
  int nz; /* number of leading zero bits in 'i' */
  int k;

  ui = i;
  nz = 32;
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
