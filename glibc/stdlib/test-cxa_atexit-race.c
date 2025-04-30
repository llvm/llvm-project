/* Bug 14333: a test for __cxa_atexit/exit race.
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

#include <stdio.h>

#define CALL_ATEXIT __cxa_atexit (&no_op, NULL, NULL)
#define CALL_EXIT exit (0)

int __cxa_atexit (void (*func) (void *), void *arg, void *d);

static void
no_op (void *ignored)
{
}

#include <stdlib/test-atexit-race-common.c>
