/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	rtc3f.c - Implements rtc subprogram.  */

#include "ent3f.h"

#include <time.h>

double ENT3F(RTC, rtc)()
{
  double elapsed;
  time_t ltime;

  time(&ltime);
  elapsed = (double)(ltime);

  return elapsed;
}
