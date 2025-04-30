/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	short3f.c - Implements short call for DFLIB  */

#include "ent3f.h"

short
ENT3F(SHORT, short)(int *input)
{
  return (short)(*input);
}
