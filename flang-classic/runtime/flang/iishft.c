/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \brief  IISHFT intrinsic  */
int
ftn_i_iishft(int i, int j)
{
  if (j > 0) {
    if (j >= 16)
      return 0;
    return (i << (j + 16)) >> 16;
  }
  if (j <= -16)
    return 0;
  return (i & 0xffff) >> -j;
}
