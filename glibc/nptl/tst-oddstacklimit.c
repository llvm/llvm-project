/* Test NPTL with stack limit that is not a multiple of the page size.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <stdlib.h>

/* This sets the stack resource limit to 1023kb, which is not a multiple
   of the page size since every architecture's page size is > 1k.  */
#ifndef ODD_STACK_LIMIT
# define ODD_STACK_LIMIT (1023 * 1024)
#endif

static const char *command;

static int
do_test (void)
{
  int ret;
  struct rlimit rlim;

  ret = getrlimit (RLIMIT_STACK, &rlim);
  if (ret != 0)
    {
      printf ("getrlimit failed: %s\n", strerror (errno));
      return 1;
    }
  rlim.rlim_cur = ODD_STACK_LIMIT;
  ret = setrlimit (RLIMIT_STACK, &rlim);
  if (ret != 0)
    {
      printf ("setrlimit failed: %s\n", strerror (errno));
      return 1;
    }
  ret = system (command);
  if (ret == -1)
    {
      printf ("system failed: %s\n", strerror (errno));
      return 1;
    }
  if (WIFEXITED (ret))
    return WEXITSTATUS (ret);
  else
    return 1;
}

#define OPT_COMMAND	10000
#define CMDLINE_OPTIONS	\
  { "command", required_argument, NULL, OPT_COMMAND },
#define CMDLINE_PROCESS	\
  case OPT_COMMAND:	\
    command = optarg;	\
    break;
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
