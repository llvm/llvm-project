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
#include <fcntl.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static void
remove_sem (int status, void *arg)
{
  sem_unlink (arg);
}


int
do_test (void)
{
  sem_t *s;
  sem_t *s2;
  pid_t pid;
  int val;

  /* Start with a clean slate and register a clean-up action.  No need to
     act if sem_unlink fails because we will catch the same problem during the
     sem_open below.  */
  sem_unlink ("/glibc-tst-sem4");
  on_exit (remove_sem, (void *) "/glibc-tst-sem4");

  s = sem_open ("/glibc-tst-sem4", O_CREAT, 0600, 1);
  if (s == SEM_FAILED)
    {
      if (errno == ENOSYS)
	{
	  puts ("sem_open not supported.  Oh well.");
	  return 0;
	}

      /* Maybe the shm filesystem has strict permissions.  */
      if (errno == EACCES)
	{
	  puts ("sem_open not allowed.  Oh well.");
	  return 0;
	}

      printf ("sem_open: %m\n");
      return 1;
    }

  /* We have the semaphore object.  Now try again with O_EXCL, this
     should fail.  */
  s2 = sem_open ("/glibc-tst-sem4", O_CREAT | O_EXCL, 0600, 1);
  if (s2 != SEM_FAILED)
    {
      puts ("2nd sem_open didn't fail");
      return 1;
    }
  if (errno != EEXIST)
    {
      puts ("2nd sem_open returned wrong error");
      return 1;
    }

  /* Check the value.  */
  if (sem_getvalue (s, &val) == -1)
    {
      puts ("getvalue failed");
      return 1;
    }
  if (val != 1)
    {
      printf ("initial value wrong: got %d, expected 1\n", val);
      return 1;
    }

  if (TEMP_FAILURE_RETRY (sem_wait (s)) == -1)
    {
      puts ("1st sem_wait failed");
      return 1;
    }

  pid = fork ();
  if (pid == -1)
    {
      printf ("fork failed: %m\n");
      return 1;
    }

  if (pid == 0)
    {
      /* Child.  */

      /* Check the value.  */
      if (sem_getvalue (s, &val) == -1)
	{
	  puts ("child: getvalue failed");
	  return 1;
	}
      if (val != 0)
	{
	  printf ("child: value wrong: got %d, expect 0\n", val);
	  return 1;
	}

      if (sem_post (s) == -1)
	{
	  puts ("child: post failed");
	  return 1;
	}
    }
  else
    {
      if (TEMP_FAILURE_RETRY (sem_wait (s)) == -1)
	{
	  puts ("2nd sem_wait failed");
	  return 1;
	}

      if (sem_getvalue (s, &val) == -1)
	{
	  puts ("parent: 2nd getvalue failed");
	  return 1;
	}
      if (val != 0)
	{
	  printf ("parent: value wrong: got %d, expected 0\n", val);
	  return 1;
	}
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
