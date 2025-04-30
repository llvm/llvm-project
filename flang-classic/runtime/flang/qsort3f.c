/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	qsort3f.c - Implements LIB3F qsort subprogram.  */

#include <stdlib.h>
#include <stdlib.h>
#include "ent3f.h"

void ENT3F(QSORT, qsort)(void *array, int *len, int *isize, int (*compar)())
{
  qsort(array, *len, *isize, compar);
}
