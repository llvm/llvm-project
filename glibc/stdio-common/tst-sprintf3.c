/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

/* Test bug #13941.  */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static int
do_test (void)
{
#if LDBL_MANT_DIG >= 106
  volatile union { long double l; long long x[2]; } u, v;
  char buf[64];
#endif
  int result = 0;

#if LDBL_MANT_DIG == 106 || LDBL_MANT_DIG == 113
# define COMPARE_LDBL(u, v) \
  ((u).l == (v).l && (u).x[0] == (v).x[0] && (u).x[1] == (v).x[1])
#else
# define COMPARE_LDBL(u, v) ((u).l == (v).l)
#endif

#define TEST_N(val, n) \
  do									   \
    {									   \
      u.l = (val);							   \
      snprintf (buf, sizeof buf, "%." #n "LgL", u.l);			   \
      if (strcmp (buf, #val) != 0)					   \
	{								   \
	  printf ("Error on line %d: %s != %s\n", __LINE__, buf, #val);	   \
	  result = 1;							   \
	}								   \
      if (sscanf (#val, "%Lg", &v.l) != 1 || !COMPARE_LDBL (u, v))	   \
	{								   \
	  printf ("Error sscanf on line %d: %." #n "Lg != %." #n "Lg\n",   \
		  __LINE__, u.l, v.l);					   \
	  result = 1;							   \
	}								   \
      /* printf ("%s %Lg %016Lx %016Lx\n", #val, u.l, u.x[0], u.x[1]); */  \
    }									   \
  while (0)

#define TEST(val) TEST_N (val,30)

#if LDBL_MANT_DIG >= 106
# if LDBL_MANT_DIG == 106
  TEST (2.22507385850719347803989925739e-308L);
  TEST (2.22507385850719397210554509863e-308L);
  TEST (2.22507385850720088902458687609e-308L);

  /* Verify precision is not lost for long doubles
     of the form +1.pN,-1.pM.  */
  TEST_N (3.32306998946228968225951765070082e+35L, 34);
# endif
  TEST (2.22507385850720138309023271733e-308L);
  TEST (2.22507385850720187715587855858e-308L);
  TEST (2.2250738585074419930597574044e-308L);
  TEST (4.45014771701440227211481959342e-308L);
  TEST (4.45014771701440276618046543466e-308L);
  TEST (4.45014771701440375431175711716e-308L);
  TEST (4.45014771701440474244304879965e-308L);
  TEST (7.12023634722304600689881138745e-307L);
  TEST (1.13923781555569064960474854133e-305L);
  TEST (1.13777777777777776389998996996L);
  TEST (1.13777777777777765287768750745L);
  TEST (20988295479420645138.2044444444L);
  TEST (20988295479420643090.2044444444L);
  TEST (2.14668699894294423266045294316e-292L);
# if LDBL_MANT_DIG == 106
  TEST (-2.35993711055432139266626434123e-292L);
  TEST (6.26323524637968345414769634658e-302L);
  TEST (1.49327164802066885331814201989e-308L);
  TEST (3.71834550652787023640837473722e-308L);
  TEST (9.51896449671134907001349268087e-306L);
# endif
#endif
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
