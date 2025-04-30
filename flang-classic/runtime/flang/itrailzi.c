/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
int
__mth_i_itrailzi(int i, int size)
{
  unsigned ui=i;  /* unsigned representation of 'i' */

  return (ui) ? __builtin_ctz(ui): (size*8);
}
