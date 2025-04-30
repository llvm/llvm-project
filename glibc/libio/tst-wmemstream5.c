/* Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <wchar.h>

extern int fclose (FILE*);

#if defined __GNUC__ && __GNUC__ >= 11
/* Verify that calling fclose on the result of open_wmemstream doesn't
   trigger GCC -Wmismatched-dealloc with fclose forward-declared and
   without <stdio.h> included in the same translation unit.  */
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wmismatched-dealloc"
#endif

static int
do_test (void)
{
  {
    wchar_t *buf;
    size_t size;
    FILE *f = open_wmemstream (&buf, &size);
    fclose (f);
  }

  {
    FILE* (*pf)(wchar_t**, size_t*) = open_wmemstream;
    wchar_t *buf;
    size_t size;
    FILE *f = pf (&buf, &size);
    fclose (f);
  }

  return 0;
}

#if defined __GNUC__ && __GNUC__ >= 11
/* Restore -Wmismatched-dealloc setting.  */
# pragma GCC diagnostic pop
#endif

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
