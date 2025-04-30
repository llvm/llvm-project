/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	commitqq3f.c - Implements DFLIB commitqq subroutine.  */

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(COMMITQQ, commitqq)(int *lu)
{
  FILE *f;
  int i;

  f = __getfile3f(*lu);
  if (f) {
    fflush(f);
    i = -1; /* .true. returned if open unit is passed */
  } else
    i = 0; /* .false.  returned if unopened unit is passed */

  return i;
}
