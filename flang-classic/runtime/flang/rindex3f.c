/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	rindex3f.c - Implements LIB3F rindex subprogram.  */

#include "ent3f.h"

int ENT3F(RINDEX, rindex)(DCHAR(a1), DCHAR(a2) DCLEN(a1) DCLEN(a2))
{
  char *a1 = CADR(a1); /* pointer to string being searched */
  char *a2 = CADR(a2); /* pointer to string being searched for */
  int a1_l = CLEN(a1); /* length of a1 */
  int a2_l = CLEN(a2); /* length of a2 */
  int i1, i2, match;

  for (i1 = a1_l - a2_l; i1 >= 0; i1--) {
    match = 1;
    for (i2 = 0; i2 < a2_l; i2++) {
      if (a1[i1 + i2] != a2[i2]) {
        match = 0;
        break;
      }
    }
    if (match)
      return (i1 + 1);
  }
  return (0);
}
