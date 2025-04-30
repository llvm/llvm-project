/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>


static int val;


static void
prepare1 (void)
{
  val *= 2;
}

static void
prepare2 (void)
{
  ++val;
}

static void
parent1 (void)
{
  val += 4;
}

static void
parent2 (void)
{
  val *= 4;
}

static void
child1 (void)
{
  val += 8;
}

static void
child2 (void)
{
  val *= 8;
}


static int
do_test (void)
{
  pid_t pid;
  int status = 0;

  if (pthread_atfork (prepare1, parent1, child1) != 0)
    {
      puts ("1st atfork failed");
      exit (1);
    }
  if (pthread_atfork (prepare2, parent2, child2) != 0)
    {
      puts ("2nd atfork failed");
      exit (1);
    }

  pid = fork ();
  if (pid == -1)
    {
      puts ("fork failed");
      exit (1);
    }

  if (pid != 0)
    {
      /* Parent.  */
      if (val != 24)
	{
	  printf ("expected val=%d, got %d\n", 24, val);
	  exit (1);
	}

      if (TEMP_FAILURE_RETRY (waitpid (pid, &status, 0)) != pid)
	{
	  puts ("waitpid failed");
	  exit (1);
	}
    }
  else
    {
      /* Child.  */
      if (val != 80)
	{
	  printf ("expected val=%d, got %d\n", 80, val);
	  exit (2);
	}
    }

  return status;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
