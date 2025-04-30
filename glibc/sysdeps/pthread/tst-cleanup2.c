/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Bao Duong <bduong@progress.com>, 2003.

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

#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

static sigjmp_buf jmpbuf;

static void
sig_handler (int signo)
{
  siglongjmp (jmpbuf, 1);
}

static int
do_test (void)
{
  char *p = NULL;
  /* gcc can overwrite the success written value by scheduling instructions
     around sprintf.  It is allowed to do this since according to C99 the first
     argument of sprintf is a character array and NULL is not a valid character
     array.  Mark the return value as volatile so that it gets reloaded on
     return.  */
  volatile int ret = 0;

  if (signal (SIGSEGV, &sig_handler) == SIG_ERR)
    {
      perror ("installing SIGSEGV handler");
      return 1;
    }

  puts ("Attempting to sprintf to null ptr");
  if (setjmp (jmpbuf))
    {
      puts ("Exiting main...");
      return ret;
    }

  sprintf (p, "This should segv\n");

  return 1;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
