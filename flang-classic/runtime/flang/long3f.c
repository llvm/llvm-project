/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	long3f.c - Implements long call for DFLIB  */

#include "ent3f.h"

int
ENT3F(LONG, long)(short *input)
{
  return (int)(*input);
}
