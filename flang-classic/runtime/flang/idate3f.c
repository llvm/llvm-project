/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	idate3f.c - Implements LIB3F idate subroutine.  */

#include "ent3f.h"

#include <time.h>

void
    ENT3F(IDATE, idate)(int *date_array)
{
  time_t ltime;
  struct tm *ltimvar;
  int yr;

  ltime = time(0);
  ltimvar = localtime(&ltime);
  date_array[0] = ltimvar->tm_mon + 1;
  date_array[1] = ltimvar->tm_mday;
  yr = ltimvar->tm_year;
  if (yr > 99)
    yr = yr % 100;
  date_array[2] = yr;
}
