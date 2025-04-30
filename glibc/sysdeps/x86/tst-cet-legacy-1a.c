/* Test for re-exec with legacy bitmap.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <support/check.h>

/* Nonzero if the program gets called via `exec'.  */
static int restart;

#define CMDLINE_OPTIONS \
  { "restart", no_argument, &restart, 1 },

extern int do_test (int argc, char *argv[]);

extern int in_dso_1 (void);
extern int in_dso_2 (void);

static int
check (void)
{
  if (in_dso_1 () != 0x1234678)
    {
      puts ("in_dso_1 () != 0x1234678");
      exit (1);
    }

  if (in_dso_2 () != 0xbadbeef)
    {
      puts ("in_dso_2 () != 0xbadbeef");
      exit (1);
    }

  return 0;
}

int
do_test (int argc, char *argv[])
{
  /* We must have
     - one or four parameters left if called initially
       + path for ld.so		optional
       + "--library-path"	optional
       + the library path	optional
       + the application name
  */

  if (restart)
    return check ();

  if (argc != 2 && argc != 5)
    FAIL_EXIT1 ("wrong number of arguments (%d)", argc);

  if (argc == 5)
    execl (argv[1], argv[1], argv[2], argv[3], argv[4], "--direct",
	   "--restart",  NULL);
  else
    execl (argv[1], argv[1], "--direct", "--restart",  NULL);

  return -1;
}

#define TEST_FUNCTION_ARGV do_test
#include <support/test-driver.c>
