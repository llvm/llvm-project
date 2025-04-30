/* Test to strtod etc for numbers like x000...0000.000e-nn.
   This file is part of the GNU C Library.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2001.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tst-strtod.h"

#define TEST_STRTOD(FSUF, FTYPE, FTOSTR, LSUF, CSUF)			\
static int								\
test_strto ## FSUF (void)						\
{									\
  char buf[300];							\
  int cnt;								\
  int result = 0;							\
									\
  for (cnt = 0; cnt < 200; ++cnt)					\
    {									\
      ssize_t n;							\
      FTYPE f;								\
									\
      n = sprintf (buf, "%d", cnt);					\
      memset (buf + n, '0', cnt);					\
      sprintf (buf + n + cnt, ".000e-%d", cnt);				\
      f = strto ## FSUF (buf, NULL);					\
									\
      if (f != (FTYPE) cnt)						\
	{								\
	  char fstr[FSTRLENMAX];					\
	  char fcntstr[FSTRLENMAX];					\
	  FTOSTR (fstr, sizeof (fstr), "%g", f);			\
	  FTOSTR (fcntstr, sizeof (fstr), "%g", (FTYPE) cnt); 		\
	  printf ("strto" #FSUF "(\"%s\") "				\
		  "failed for cnt == %d (%s instead of %s)\n",		\
		  buf, cnt, fstr, fcntstr);				\
	  result = 1;							\
	}								\
      else								\
	printf ( "strto" #FSUF "() fine for cnt == %d\n", cnt);		\
    }									\
  return result;							\
}

GEN_TEST_STRTOD_FOREACH (TEST_STRTOD)

int
main (void)
{
  return STRTOD_TEST_FOREACH (test_strto);
}
