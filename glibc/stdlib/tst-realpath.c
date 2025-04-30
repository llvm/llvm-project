/* Test to verify that realpath() doesn't cause false positives due
   to GCC attribute malloc.

   Test failure exposes the presence of the attribute in the following
   declaration:

   __attribute__ ((__malloc__ (free, 1))) char*
   realpath (const char *, char *);

   Copyright (C) 2021 Free Software Foundation, Inc.
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
#include <malloc.h>

#if defined __GNUC__ && __GNUC__ >= 11
/* Turn GCC -Wmismatched-dealloc warnings into errors to expose false
   positives.  */
#  pragma GCC diagnostic push
#  pragma GCC diagnostic error "-Wmismatched-dealloc"

/* Associate dealloc as the only deallocator suitable for pointers
   returned from alloc.
   GCC automatically disables inlining of allocator and deallocator
   functions marked with the argument form of attribute malloc but
   it doesn't hurt to disable it explicitly.  */
__attribute  ((noipa)) void dealloc (void *);
__attribute ((malloc (dealloc, 1))) char* alloc (void);
#endif

void dealloc (void *p)
{
  free (p);
}

char* alloc (void)
{
  return (char *)malloc (8);
}

static int
do_test (void)
{
  char *resolved_path = alloc ();
  char *ret = realpath ("/", resolved_path);
  dealloc (ret);

  resolved_path = alloc ();
  ret = realpath ("/", resolved_path);
  dealloc (resolved_path);

  /* The following should emit a warning (but doesn't with GCC 11):
     resolved_path = alloc ();
     ret = realpath ("/", resolved_path);
     free (ret);   // expect -Wmismatched-dealloc
  */

  return 0;
}

#if defined __GNUC__ && __GNUC__ >= 11
/* Restore -Wmismatched-dealloc setting.  */
#  pragma GCC diagnostic pop
#endif

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
