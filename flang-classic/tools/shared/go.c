/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "gbldefs.h"
#include "global.h"
#include "go.h"
#include <stdio.h>

#if DEBUG + 0
#include <stdarg.h>

void
dbgprintf(const char *format, ...)
{
  if ((flg.dbg[64] & 1) || flg.dbg[65] > 0) {
    FILE *f = gbl.dbgfil ? gbl.dbgfil : stderr;
    va_list va;
    va_start(va, format);
    vfprintf(f, format, va);
    va_end(va);
  }
}

/*
 * yes(), yes_silent(), and no() are returned by go_odometer() to effect
 * debug output and return a particular Boolean value.
 */

/* do vfprintf() and return true */
static int
yes(const char *format, ...)
{
  FILE *f = gbl.dbgfil ? gbl.dbgfil : stderr;
  va_list va;
  va_start(va, format);
  vfprintf(f, format, va);
  va_end(va);
  fputc('\n', f);
  return true;
}

/* no output, just return true */
static int
yes_silent(const char *format, ...)
{
  return true;
}

/* do vfprintf() and return false */
static int
no(const char *format, ...)
{
  FILE *f = gbl.dbgfil ? gbl.dbgfil : stderr;
  va_list va;
  va_start(va, format);
  vfprintf(f, format, va);
  va_end(va);
  fputc('\n', f);
  return false;
}

/* Implementation of the go() macro, which passes __FILE__ and __LINE__
 * to each call.
 */
int (*go_odometer(const char *file, int line))(const char *format, ...)
{
  if ((flg.dbg[64] & 1) || flg.dbg[65] > 0) {
    static int odometer = 0;
    if (++odometer < flg.dbg[65] || flg.dbg[65] <= 0) {
      if (odometer + 1 == flg.dbg[65])
        dbgprintf("[%d LAST!]%s(%d): ", odometer, file, line);
      else
        dbgprintf("[%d]%s(%d): ", odometer, file, line);
      return yes;
    } else {
      dbgprintf("[%d NO]%s(%d): ", odometer, file, line);
      return no;
    }
  } else {
    return yes_silent;
  }
}

#endif /* DEBUG && __STDC__ */
