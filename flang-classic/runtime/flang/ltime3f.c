/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	ltime3f.c - Implements LIB3F ltime subprogram.  */

#include "ent3f.h"

typedef struct {
  int m[9]; /* 9 elements in tm structure */
} _TM;

/*
 * extern struct tm *localtime(const time_t *);
 *  the argument is either a pointer to 32-bit or 64-bit int depending on
 *  sizeof(time_t)
 */
extern _TM *localtime(void *);

void ENT3F(LTIME, ltime)(void *stime, _TM *tarray)
{
  _TM *p;
  p = localtime(stime);
  *tarray = *p;
}
