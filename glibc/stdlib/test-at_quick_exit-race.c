/* Bug 14333: a test for at_quick_exit/quick_exit race.

   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* This file must be run from within a directory called "stdlib".  */

/* See stdlib/test-atexit-race-common.c for details on this test.  */

#define CALL_ATEXIT at_quick_exit (&no_op)
#define CALL_EXIT quick_exit (0)

static void
no_op (void)
{
}

#include <stdlib/test-atexit-race-common.c>
