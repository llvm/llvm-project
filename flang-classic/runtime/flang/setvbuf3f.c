/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* this isn't an actual 3f routine.  But it is useful */

/* setvbuf3f(lu,type,size)

   lu is the logical unit
   type is 0 - full buffering, 1 - line buffering, 2 - no buffering
   size is the size of the new buffer

   it returns 0 on success, non-zero on error
*/

#include <stdio.h>
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(SETVBUF3F, setvbuf3f)(int *lu, int *type, int *size)
{
  FILE *f;
  int t;

  f = __getfile3f(*lu);
  if (f == NULL) {
    return (1);
  }
  if (*type == 0) {
    t = _IOFBF;
  } else if (*type == 1) {
    t = _IOLBF;
  } else if (*type == 2) {
    t = _IONBF;
  } else {
    return (1);
  }
  if (setvbuf(f, NULL, t, *size) != 0) {
    return (1);
  }
  return (0);
}

int ENT3F(SETVBUF, setvbuf)(int *lu, int *type, int *size,
                            DCHAR(buf) DCLEN(buf))
{
  FILE *f;
  int t;

  f = __getfile3f(*lu);
  if (f == NULL) {
    return (1);
  }
  if (*type == 0) {
    t = _IOFBF;
  } else if (*type == 1) {
    t = _IOLBF;
  } else if (*type == 2) {
    t = _IONBF;
  } else {
    return (1);
  }
  if (setvbuf(f, CADR(buf), t, *size) != 0) {
    return (1);
  }
  return (0);
}
