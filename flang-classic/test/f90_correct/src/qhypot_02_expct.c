/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Part of the F2008 HYPOT intrinsic test with quad-precision arguments
 */

#include <stdio.h>
#include <math.h>

void
get_expected_q(long double src1[], long double src2[], long double expct[],
               int n)
{
  int i;

  for (i = 0; i < n; i++) {
    expct[i] = hypotl(src1[i], src2[i]);
  }
}
