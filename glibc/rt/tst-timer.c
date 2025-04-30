/* Tests for POSIX timer implementation.  Dummy version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <unistd.h>

/* This file is only used if there is no other implementation and it should
   means that there is no implementation of POSIX timers.  */
static int
do_test (void)
{
#ifdef _POSIX_TIMERS
  /* There should be a test.  */
  return 1;
#else
  return 0;
#endif
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
