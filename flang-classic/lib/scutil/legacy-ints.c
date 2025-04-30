/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief Convert 64-bit ints to/from legacy big-endian 2-word form.
 */

#include "legacy-ints.h"

void
bgitoi64(int64_t v, DBLINT64 res)
{
  res[0] = v >> 32;
  res[1] = v;
}

int64_t
i64tobgi(DBLINT64 v)
{
  int64_t x = ((int64_t) v[0] << 32) | (uint32_t) v[1];
  return x;
}
