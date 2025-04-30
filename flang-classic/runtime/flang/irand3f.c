/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	irand3f.c - Implements LIB3F irand subprogram.  */

#include "ent3f.h"

extern int rand();
#if defined(_WIN64)
int ENT3F(IRAND1, irand1)() { return rand(); }
int ENT3F(IRAND2, irand2)(int *flag)
{
  if (*flag)
    srand(*flag);
  return rand();
}

#else
int ENT3F(IRAND, irand)() { return rand(); }
#endif
