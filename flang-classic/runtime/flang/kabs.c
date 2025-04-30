/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdint.h>

int64_t
ftn_i_kabs(int64_t i0)
{
  if (i0 >= 0)
    return i0;
  return -i0;
}
