/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* fortran miscellaneous support routines */

#include <time.h>
#include <math.h>
#include <stdio.h>
#include "enames.h"

typedef int INT;

static int
yr2(int yr)
{
  int y = yr;
  if (y > 99)
    y = y % 100;
  return y;
}

/* ***********************************************************************/
/* function: Ftn_date:
 *
 * buffer receives date in the form dd-mmm-yy. If the length of buf is not 9
 * string is either truncated or blank filled.
 */
/* ***********************************************************************/

void
Ftn_date(char *buf,     /* date buffer */
         INT buf_len)   /* length of date buffer */
{
  char loc_buf[10];
  INT idx1;
  time_t ltime;
  struct tm *lt;
  static const char *month[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

  /* procedure */

  ltime = time(0);
  lt = localtime(&ltime);
  sprintf(loc_buf, "%2d-%3s-%02d", lt->tm_mday, month[lt->tm_mon],
          yr2(lt->tm_year));

  for (idx1 = 0; idx1 < buf_len; idx1++) {
    if (idx1 > 8)
      buf[idx1] = ' ';
    else
      buf[idx1] = loc_buf[idx1];
  }
}
/* ***********************************************************************/
/* function: Ftn_datew:
 *
 * buffer receives date in the form dd-mmm-yy.
 */
/* ***********************************************************************/

void
Ftn_datew(char buf[9]) /* date buffer */
{

  /* procedure */

  Ftn_date(buf, 9);
}
/* ***********************************************************************/
/* function: Ftn_jdate:
 *
 * i,j,k receive integer values for month, day, and year
 */
/* ***********************************************************************/

void
Ftn_jdate(INT *i, /* month */
          INT *j, /* day */
          INT *k) /* year */
{
  time_t ltime;
  struct tm *ltimvar;

  /* procedure */

  ltime = time(0);
  ltimvar = localtime(&ltime);
  *i = ltimvar->tm_mon + 1;
  *j = ltimvar->tm_mday;
  *k = yr2(ltimvar->tm_year);
}
/* ***********************************************************************/
/* function: Ftn_idate:
 *
 * i,j,k receive integer values for month, day, and year
 */
/* ***********************************************************************/

void
Ftn_idate(short *i, /* month */
          short *j, /* day */
          short *k) /* year */
{
  INT loc_i, loc_j, loc_k;

  /* procedure */

  Ftn_jdate(&loc_i, &loc_j, &loc_k);
  *i = (short)loc_i;
  *j = (short)loc_j;
  *k = (short)loc_k;
}
/* ***********************************************************************/
/* function: Ftn_secnds:
 *
 * Returns the number of real time seconds since midnight minus the supplied
 * value
 */
/* ***********************************************************************/

float
Ftn_secnds(float x)
{
  static int called = 0;
  static int diffs;
  int i;
  time_t ltime;
  struct tm *lt;
  float f;

  ltime = time(0);
  if (called == 0) {
    called = 1; /* first time called */
                /*
                 * compute value to subtract from time(0) to give seconds since
                 * midnight
                 */
    lt = localtime(&ltime);
    i = lt->tm_sec + (60 * lt->tm_min) + (3600 * lt->tm_hour);
    diffs = ltime - i;
  }
  f = (float)(ltime - diffs);
  return (f - x);
}
/* ***********************************************************************/
/* function: Ftn_dsecnds:
 *
 * double precision version of secnds.
 */
/* ***********************************************************************/

double
Ftn_dsecnds(double x)
{
  static int called = 0;
  static int diffs;
  int i;
  time_t ltime;
  struct tm *lt;
  double f;

  ltime = time(0);
  if (called == 0) {
    called = 1; /* first time called */
                /*
                 * compute value to subtract from time(0) to give seconds since
                 * midnight
                 */
    lt = localtime(&ltime);
    i = lt->tm_sec + (60 * lt->tm_min) + (3600 * lt->tm_hour);
    diffs = ltime - i;
  }
  f = (double)(ltime - diffs);
  return (f - x);
}
/* ***********************************************************************/
/* function: Ftn_time:
 *
 * buf returns time in the form hh:mm:ss. If the length of buf is not 8,
 * the string will be either truncated or blank filled.
 */
/* ***********************************************************************/
void
Ftn_time(char *buf,   /* time buffer */
         INT buf_len) /* length of buffer */
{
  char loc_buf[10];
  INT idx1;
  time_t ltime;
  struct tm *ltimvar;

  /* procedure */

  ltime = time(0);
  ltimvar = localtime(&ltime);
  (void)sprintf(&loc_buf[0], "%2.2d", ltimvar->tm_hour);
  (void)sprintf(&loc_buf[3], "%2.2d", ltimvar->tm_min);
  (void)sprintf(&loc_buf[6], "%2.2d", ltimvar->tm_sec);
  loc_buf[2] = ':';
  loc_buf[5] = ':';
  for (idx1 = 0; idx1 < buf_len; idx1++) {
    if (idx1 > 7)
      buf[idx1] = ' ';
    else
      buf[idx1] = loc_buf[idx1];
  }
}

/* ***********************************************************************/
/* function: Ftn_timew:
 *
 * buf returns time in the form hh:mm:ss.
 */
/* ***********************************************************************/
void
Ftn_timew(char buf[8]) /* time buffer */
{
  Ftn_time(buf, 8);
}
/* ***********************************************************************/
/* function: Ftn_ran (Ftn_dran - double precision version of ran):
 *
 * Ftn_ran returns VMS-compatible random number sequence
 */
/* ***********************************************************************/
float
Ftn_ran(unsigned *seed)
{
  static unsigned tm24 = 0x33800000; /* 2 ** -24 */
  unsigned u;

  u = *seed = *seed * 69069 + 1;
  /*
   * extract higher-order 24 bits of new seed, convert to float, and
   * scale converted value so that  0.0 <= value < 1.0.
   */
  return (((float)(u >> 8)) * *(float *)&tm24);
}

double
Ftn_dran(unsigned *seed)
{
  return (Ftn_ran(seed));
}

/* ***********************************************************************/
#if defined(TARGET_WIN)
/*
 * Miscellaneous support routines for windows '3f-like' routines
 * which are self-contained as opposed to fortran interfaces to
 * C system routines.
 */
void
CopyMemory(char *to, char *from, size_t n)
{
  size_t i;
  for (i = 0; i < n; i++)
    to[i] = from[i];
  return;
}

int
MakeWord(int lo, int hi)
{
  return ((hi & 0xff) << 8) | (lo & 0xff);
}
#endif
