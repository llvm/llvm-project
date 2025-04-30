/* Test that gettext() in multithreaded applications works correctly.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2008.

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

#include <libintl.h>
#include <locale.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

pthread_barrier_t b;

static void *
tf (void *arg)
{
  pthread_barrier_wait (&b);
  return gettext ("Operation not permitted");
}

int
test (void)
{
  pthread_t th[4];
  unsetenv ("LANGUAGE");
  unsetenv ("OUTPUT_CHARSET");
  textdomain ("tstgettext6");
  bindtextdomain ("tstgettext6", OBJPFX "domaindir");
  setlocale (LC_ALL, "ja_JP.UTF-8");
  pthread_barrier_init (&b, NULL, 4);
  for (int i = 0; i < 4; i++)
    if (pthread_create (&th[i], NULL, tf, NULL))
      {
	puts ("pthread_create failed");
	return 1;
      }
  for (int i = 0; i < 4; i++)
    pthread_join (th[i], NULL);
  return 0;
}

int
main (void)
{
  for (int i = 0; i < 300; i++)
    {
      pid_t p = fork ();
      if (p == -1)
	{
	  printf ("fork failed: %m\n");
	  return 1;
	}
      if (p == 0)
	_exit (test ());
      int status;
      wait (&status);
      if (WIFEXITED (status) && WEXITSTATUS (status) != 0)
	{
	  printf ("child exited with %d\n", WEXITSTATUS (status));
	  return 1;
	}
      else if (WIFSIGNALED (status))
	{
	  printf ("child killed by signal %d\n", WTERMSIG (status));
	  return 1;
	}
    }
  return 0;
}
