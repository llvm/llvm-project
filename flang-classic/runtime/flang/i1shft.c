/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**\brief  ISHFT(integer*1) intrinsic  */
int
ftn_i_i1shft(int i, int j)
{
  if (j > 0) {
    if (j >= 8)
      return 0;
    return (i << (j + 24)) >> 24;
  }
  if (j <= -8)
    return 0;
  return (i & 0xff) >> -j;
}
