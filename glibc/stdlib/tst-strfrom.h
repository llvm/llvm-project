/* Tests for strfromf, strfromd, strfroml functions.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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
#include <float.h>
#include <math.h>
#include <locale.h>

#include "tst-strtod.h"

#define _CONCAT(a, b) a ## b
#define CONCAT(a, b) _CONCAT (a, b)

/* Generator to create an FTYPE member variabled named FSUF
 *    used to populate struct member variables.  */
#define FTYPE_MEMBER(FSUF, FTYPE, FTOSTR, LSUF, CSUF)  \
       FTYPE FSUF;

#define STRUCT_FOREACH_FLOAT_FTYPE GEN_TEST_STRTOD_FOREACH (FTYPE_MEMBER)

#define ENTRY(FSUF, FTYPE, FTOSTR, LSUF, CSUF, ...)  \
   CONCAT (__VA_ARGS__, LSUF),
/* This is hacky way around the seemingly unavoidable macro
 * expansion of the INFINITY or HUGE_VAL like macros in the
 * above.  It is assumed the compiler will implicitly convert
 * the infinity correctly.  */
#define INF INFINITY + 0.0
#define NAN_ NAN + 0.0

struct test_input
{
  STRUCT_FOREACH_FLOAT_FTYPE
};
struct test {
  const char *s;
  const char *fmt;
  int size;
  int rc;
  struct test_input t;
};
#define TEST(s, fmt, size, rc, val)				\
  {								\
    s, fmt, size, rc, { GEN_TEST_STRTOD_FOREACH (ENTRY, val) }	\
  }
/* Hexadecimal tests.  */
struct htests
{
  const char *fmt;
  const char *exp[4];
  struct test_input t;
};
#define HTEST(fmt, exp1, exp2, exp3, exp4, val)				  \
  {									  \
    fmt, exp1, exp2, exp3, exp4, { GEN_TEST_STRTOD_FOREACH (ENTRY, val) } \
  }

#define TEST_STRFROM(FSUF, FTYPE, FTOSTR, LSUF, CSUF)			\
static int								\
test_ ## FSUF (void)							\
{									\
  char buf[50], sbuf[5];						\
  int status = 0;							\
  int i, rc = 0, rc1 = 0;						\
  for (i = 0; i < sizeof (stest) / sizeof (stest[0]); i++)		\
    {									\
      rc = FTOSTR (sbuf, stest[i].size, stest[i].fmt, stest[i].t.FSUF);	\
      rc1 = (strcmp (sbuf, stest[i].s) != 0) || (rc != stest[i].rc);	\
      if (rc1)								\
	{								\
	  printf (#FTOSTR ": got %s (%d), expected %s (%d)\n",		\
		  sbuf, rc, stest[i].s, stest[i].rc);			\
	  status++;							\
	}								\
    }									\
  for (i = 0; i < sizeof (tests) / sizeof (tests[0]); i++)		\
    {									\
      rc = FTOSTR (buf, tests[i].size, tests[i].fmt, tests[i].t.FSUF);	\
      rc1 = (strcmp (buf, tests[i].s) != 0) || (rc != tests[i].rc);	\
      if (rc1)								\
	{								\
	  printf (#FTOSTR ": got %s (%d), expected %s (%d)\n",		\
		  buf, rc, tests[i].s, tests[i].rc);			\
	  status++;							\
	}								\
    }									\
  for (i = 0; i < sizeof (htest) / sizeof (htest[0]); i++)		\
    {									\
      rc = FTOSTR (buf, 50, htest[i].fmt, htest[i].t.FSUF);		\
      if (strcmp (buf, htest[i].exp[0]) == 0				\
	  || strcmp (buf, htest[i].exp[1]) == 0				\
	  || strcmp (buf, htest[i].exp[2]) == 0				\
	  || strcmp (buf, htest[i].exp[3]) == 0)			\
	continue;							\
      else								\
	{								\
	  printf (#FTOSTR ": got %s (%d), expected %s or %s or %s "	\
		  "or %s\n", buf, rc, htest[i].exp[0], htest[i].exp[1],	\
		  htest[i].exp[2], htest[i].exp[3]);			\
	  status++;							\
	}								\
    }									\
  return status;							\
}
