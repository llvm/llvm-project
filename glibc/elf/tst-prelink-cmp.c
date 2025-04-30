/* Test the output from the environment variable, LD_TRACE_PRELINKING,
   for prelink.
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

#include <ldsodefs.h>
#include <support/check.h>
#include <support/xstdio.h>
#include <support/support.h>
#include <support/test-driver.h>

static int
do_test (void)
{
#ifndef DL_EXTERN_PROTECTED_DATA
  return EXIT_UNSUPPORTED;
#else
  char *src = xasprintf ("%s/elf/tst-prelink-conflict.out",
                         support_objdir_root);
  FILE *f = xfopen (src,"r");
  size_t buffer_length = 0;
  char *buffer = NULL;

  const char *expected = "/0 stdout\n";

  xgetline (&buffer, &buffer_length, f);
  TEST_COMPARE_STRING (expected, buffer);

  free (buffer);
  xfclose (f);
  return 0;
#endif
}

#include <support/test-driver.c>
