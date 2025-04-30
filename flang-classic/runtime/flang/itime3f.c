/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	itime3f.c - Implements LIB3F itime subroutine.  */

#include "ent3f.h"

#include <time.h>

void ENT3F(ITIME, itime)(int iarray[3])
{
  time_t ltime;
  struct tm *ltimvar;

  ltime = time(0);
  ltimvar = localtime(&ltime);
  iarray[0] = ltimvar->tm_hour;
  iarray[1] = ltimvar->tm_min;
  iarray[2] = ltimvar->tm_sec;
  return;
}
