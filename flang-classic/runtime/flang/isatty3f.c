/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	isatty3f.c - Implements LIB3F isatty subprogram.  */

#include "ent3f.h"
#include "utils3f.h"

int ENT3F(ISATTY, isatty)(int *lu)
{
  if (__isatty3f(*lu))
    return -1; /* .true. */
  return 0;
}
