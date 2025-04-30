/* This file is part of the GNU C Library.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
   Contributed by Marek Polacek <polacek@redhat.com>, 2012.

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

#include <array_length.h>
#include <wchar.h>

/* Prototype for our test function.  */
static int do_test (void);

static int
do_test (void)
{
#ifndef NO_LONG_DOUBLE
  int result = 0;
  const long double x = 24.5;
  wchar_t a[16];
  swprintf (a, array_length (a), L"%La\n", x);
  wchar_t A[16];
  swprintf (A, array_length (a), L"%LA\n", x);

  /* Here wprintf can return four valid variants.  We must accept all
     of them.  */
  result |= (wmemcmp (a, L"0xc.4p+1", 8) == 0
	     && wmemcmp (A, L"0XC.4P+1", 8) == 0);
  result |= (wmemcmp (a, L"0x3.1p+3", 8) == 0
	     && wmemcmp (A, L"0X3.1P+3", 8) == 0);
  result |= (wmemcmp (a, L"0x6.2p+2", 8) == 0
	     && wmemcmp (A, L"0X6.2P+2", 8) == 0);
  result |= (wmemcmp (a, L"0x1.88p+4", 8) == 0
	     && wmemcmp (A, L"0X1.88P+4", 8) == 0);

  return result != 1;
#else
  return 0;
#endif
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
