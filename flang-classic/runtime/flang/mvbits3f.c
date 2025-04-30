/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	mvbits3f.c - Implements LIB3F mvbits subprogram.  */

#include "ent3f.h"
#include "enames.h"
#include "ftnbitsup.h"

void ENT3F(MVBITS, mvbits)(int *src,  /* source field */
                           int *pos,  /* start position in source field */
                           int *len,  /* number of bits to move */
                           int *dest, /* destination field */
                           int *posd) /* start position in dest field */
{
  Ftn_jmvbits(*src, *pos, *len, dest, *posd);
}
