/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* INT */
int
__mth_i_idnint(double d)
{
  if (d > 0)
    return (int)(d + 0.5);
  else
    return (int)(d - 0.5);
}
