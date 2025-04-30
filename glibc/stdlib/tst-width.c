/* Test integer width macros.
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

#include <limits.h>
#include <stdio.h>

#define CHECK_WIDTH(TYPE, MAX, WIDTH)					\
  do									\
    {									\
      if ((MAX >> ((TYPE) -1 < 0 ? (WIDTH - 2) : (WIDTH - 1))) != 1)	\
	{								\
	  puts ("bad width of " #TYPE);					\
	  result = 1;							\
	}								\
      else								\
	puts ("width of " #TYPE " OK");					\
    }									\
  while (0)

static int
do_test (void)
{
  int result = 0;
#ifndef CHAR_WIDTH
# error "missing CHAR_WIDTH"
#endif
  CHECK_WIDTH (char, CHAR_MAX, CHAR_WIDTH);
#ifndef SCHAR_WIDTH
# error "missing SCHAR_WIDTH"
#endif
  CHECK_WIDTH (signed char, SCHAR_MAX, SCHAR_WIDTH);
#ifndef UCHAR_WIDTH
# error "missing UCHAR_WIDTH"
#endif
  CHECK_WIDTH (unsigned char, UCHAR_MAX, UCHAR_WIDTH);
#ifndef SHRT_WIDTH
# error "missing SHRT_WIDTH"
#endif
  CHECK_WIDTH (signed short, SHRT_MAX, SHRT_WIDTH);
#ifndef USHRT_WIDTH
# error "missing USHRT_WIDTH"
#endif
  CHECK_WIDTH (unsigned short, USHRT_MAX, USHRT_WIDTH);
#ifndef INT_WIDTH
# error "missing INT_WIDTH"
#endif
  CHECK_WIDTH (signed int, INT_MAX, INT_WIDTH);
#ifndef UINT_WIDTH
# error "missing UINT_WIDTH"
#endif
  CHECK_WIDTH (unsigned int, UINT_MAX, UINT_WIDTH);
#ifndef LONG_WIDTH
# error "missing LONG_WIDTH"
#endif
  CHECK_WIDTH (signed long, LONG_MAX, LONG_WIDTH);
#ifndef ULONG_WIDTH
# error "missing ULONG_WIDTH"
#endif
  CHECK_WIDTH (unsigned long, ULONG_MAX, ULONG_WIDTH);
#ifndef LLONG_WIDTH
# error "missing LLONG_WIDTH"
#endif
  CHECK_WIDTH (signed long long, LLONG_MAX, LLONG_WIDTH);
#ifndef ULLONG_WIDTH
# error "missing ULLONG_WIDTH"
#endif
  CHECK_WIDTH (unsigned long long, ULLONG_MAX, ULLONG_WIDTH);
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
