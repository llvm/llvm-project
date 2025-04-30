/* Tester for calling inline string functions.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Make sure we test the optimized inline functions.  */
#define __USE_STRING_INLINES	1

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <fcntl.h>


int
do_test (void)
{
  int status;
  int errors = 0;
  char buf1[1000];
  char *cp;
  char ch;

  cp = strcpy (buf1, "hello world");
  if (strcmp ("hello world", cp++) != 0)
    {
      puts ("strcmp test 1 failed");
      ++errors;
    }

  cp = buf1;
  if (strcmp (cp++, "hello world") != 0)
    {
      puts ("strcmp test 2 failed");
      ++errors;
    }

  ch = 'h';
  if (strchr ("hello world", ch++) == NULL)
    {
      puts ("strchr test 1 failed");
      ++errors;
    }

  const char * const hw = "hello world";
  if (strpbrk (hw, "o") - hw != 4)
    {
      puts ("strpbrk test 1 failed");
      ++errors;
    }

  if (errors == 0)
    {
      status = EXIT_SUCCESS;
      puts ("No errors.");
    }
  else
    {
      status = EXIT_FAILURE;
      printf ("%d errors.\n", errors);
    }
  return status;
}

#include <support/test-driver.c>
