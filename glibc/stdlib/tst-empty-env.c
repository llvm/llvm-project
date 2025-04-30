/* Test that passing a NULL value does not hang environment traversal in
   tunables.
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

/* The test is useful only when the source is configured with
   --enable-hardcoded-path-in-tests since otherwise the execve just picks up
   the system dynamic linker.  */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>

static int
do_test (int argc, char **argv)
{
  if (argc == 2)
    return 0;

  char envname[] = "FOOBAR";
  char *filename = program_invocation_name;
  char *newargv[] = {filename, filename, NULL};
  char *newenviron[] = {envname, NULL};

   /* This was reported in Fedora:

      https://bugzilla.redhat.com/show_bug.cgi?id=1414589

      If one of the environment variables has no value, then the environment
      traversal must skip and also advance to the next environment entry.  The
      bug in question would cause this test to hang in an infinite loop.  */
  int ret = execve (filename, newargv, newenviron);

  if (ret != 0)
    printf ("execve failed: %m");

  /* We will reach here only if we fail execve.  */
  return 1;
}

#define TEST_FUNCTION_ARGV do_test
#include <support/test-driver.c>
