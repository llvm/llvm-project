/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "float128.h"

extern long double roundl(long double);

float128_t __mth_i_qround(float128_t x)
{
  return roundl(x);
}
