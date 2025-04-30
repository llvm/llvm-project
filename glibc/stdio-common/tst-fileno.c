/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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
#include <unistd.h>

static int
check (const char *name, FILE *stream, int fd)
{
  int sfd = fileno (stream);
  printf ("(fileno (%s) = %d) %c= %d\n", name, sfd,
	  sfd == fd ? '=' : '!', fd);
  return sfd != fd;
}

static int
do_test (void)
{
  return (check ("stdin", stdin, STDIN_FILENO)
	  || check ("stdout", stdout, STDOUT_FILENO)
	  || check ("stderr", stderr, STDERR_FILENO));
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
