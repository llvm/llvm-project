/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Roland McGrath <roland@redhat.com>, 2002.

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


static pid_t initial_pid;


static void *
tf (void *arg)
{
  if (getppid () != initial_pid)
    {
      printf ("getppid in thread returned %ld, expected %ld\n",
	      (long int) getppid (), (long int) initial_pid);
      return (void *) -1;
    }

  return NULL;
}


int
main (void)
{
  initial_pid = getpid ();

  pid_t child = fork ();
  if (child == 0)
    {
      if (getppid () != initial_pid)
	{
	  printf ("first getppid returned %ld, expected %ld\n",
		  (long int) getppid (), (long int) initial_pid);
	  exit (1);
	}

      pthread_t th;
      if (pthread_create (&th, NULL, tf, NULL) != 0)
	{
	  puts ("pthread_create failed");
	  exit (1);
	}

      void *result;
      if (pthread_join (th, &result) != 0)
	{
	  puts ("pthread_join failed");
	  exit  (1);
	}

      exit (result == NULL ? 0 : 1);
    }
  else if (child == -1)
    {
      puts ("initial fork failed");
      return 1;
    }

  int status;
  if (TEMP_FAILURE_RETRY (waitpid (child, &status, 0)) != child)
    {
      printf ("waitpid failed: %m\n");
      return 1;
    }

  return status;
}
