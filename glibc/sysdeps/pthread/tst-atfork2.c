/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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

#include <dlfcn.h>
#include <errno.h>
#include <mcheck.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>


/* Must be exported.  */
int val;

static void
prepare (void)
{
  val *= 2;
}

static void
parent (void)
{
  val += 4;
}

static void
child (void)
{
  val += 8;
}


static int
do_test (void)
{
  mtrace ();

  if (pthread_atfork (prepare, parent, child) != 0)
    {
      puts ("do_test: atfork failed");
      exit (1);
    }

  void *h = dlopen ("tst-atfork2mod.so", RTLD_LAZY);
  if (h == NULL)
    {
      printf ("dlopen failed: %s\n", dlerror ());
      exit (1);
    }

  /* First trial of fork.  */
  pid_t pid = fork ();
  if (pid == -1)
    {
      puts ("1st fork failed");
      exit (1);
    }

  if (pid == 0)
    {
      /* Child.  */
      if (val != 80)
	{
	  printf ("1st: expected val=%d, got %d\n", 80, val);
	  exit (2);
	}

      exit (0);
    }

  /* Parent.  */
  if (val != 24)
    {
      printf ("1st: expected val=%d, got %d\n", 24, val);
      exit (1);
    }

  int status;
  if (TEMP_FAILURE_RETRY (waitpid (pid, &status, 0)) != pid)
    {
      puts ("1st waitpid failed");
      exit (1);
    }

  if (status != 0)
    exit (status);

  puts ("unloading now");

  /* Unload the module.  */
  if (dlclose (h) != 0)
    {
      puts ("dlclose failed");
      exit (1);
    }

  puts ("2nd fork");

  /* Second fork trial.   */
  val = 1;
  pid = fork ();
  if (pid == -1)
    {
      puts ("2nd fork failed");
      exit (1);
    }

  if (pid == 0)
    {
      /* Child.  */
      if (val != 10)
	{
	  printf ("2nd: expected val=%d, got %d\n", 10, val);
	  exit (3);
	}

      exit (0);
    }

  /* Parent.  */
  if (val != 6)
    {
      printf ("2nd: expected val=%d, got %d\n", 6, val);
      exit (1);
    }

  if (TEMP_FAILURE_RETRY (waitpid (pid, &status, 0)) != pid)
    {
      puts ("2nd waitpid failed");
      exit (1);
    }

  if (status != 0)
    exit (status);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
