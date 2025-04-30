/* Test fortify-source allocation size handling in pvalloc (bug 25401).
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#undef _FORTIFY_SOURCE
#define _FORTIFY_SOURCE 2
#include <malloc.h>
#include <string.h>
#include <support/check.h>
#include <support/xunistd.h>
#include <unistd.h>

static int
do_test (void)
{
  /* The test below assumes that pvalloc rounds up the allocation size
     to at least 8.  */
  TEST_VERIFY (xsysconf (_SC_PAGESIZE) >= 8);

  void *p = pvalloc (5);
  TEST_VERIFY_EXIT (p != NULL);

  /* This is valid assuming the page size is at least 8 because
     pvalloc rounds up the allocation size to a multiple of the page
     size.  Due to bug 25041, this used to trigger a compiler
     warning.  */
  strcpy (p, "abcdefg");

  asm ("" : : "g" (p) : "memory"); /* Optimization barrier.  */
  TEST_VERIFY (malloc_usable_size (p) >= xsysconf (_SC_PAGESIZE));
  return 0;
}

#include <support/test-driver.c>
