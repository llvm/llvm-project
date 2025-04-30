/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2004.

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
#include <string.h>
#include <wchar.h>

static int
do_test (void)
{
  FILE *f = popen ("echo test", "r");
  int result = 0, ret;
  char *line = NULL;
  size_t len = 0;

  if (f == NULL)
    {
      printf ("popen failed %m");
      return 1;
    }

  /* POSIX says that pipe streams are byte-oriented.  */
  if (fwide (f, 0) >= 0)
    {
      puts ("popen did not return byte-oriented stream");
      result = 1;
    }

  if (getline (&line, &len, f) != 5)
    {
      puts ("could not read line from popen");
      result = 1;
    }
  else if (strcmp (line, "test\n") != 0)
    {
      printf ("read \"%s\"\n", line);
      result = 1;
    }

  if (getline (&line, &len, f) != -1)
    {
      puts ("second getline did not return -1");
      result = 1;
    }

  ret = pclose (f);
  if (ret != 0)
    {
      printf ("pclose returned %d\n", ret);
      result = 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
