/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdint.h>

int64_t
ftn_i_kisign(int64_t i, int64_t j)
{
  int64_t absi;

  absi = i >= 0 ? i : -i;
  if (j >= 0)
    return absi;
  return -absi;
}
