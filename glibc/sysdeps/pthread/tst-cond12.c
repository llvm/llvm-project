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

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>


static char fname[] = "/tmp/tst-cond12-XXXXXX";
static int fd;


static void prepare (void);
#define PREPARE(argc, argv) prepare ()

static int do_test (void);
#define TEST_FUNCTION do_test ()

#include "../test-skeleton.c"


static void
prepare (void)
{
  fd = mkstemp (fname);
  if (fd == -1)
    {
      printf ("mkstemp failed: %m\n");
      exit (1);
    }
  add_temp_file (fname);
  if (ftruncate (fd, 1000) < 0)
    {
      printf ("ftruncate failed: %m\n");
      exit (1);
    }
}


static int
do_test (void)
{
  struct
  {
    pthread_mutex_t m;
    pthread_cond_t c;
    int var;
  } *p = mmap (NULL, sizeof (*p), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED)
    {
      printf ("initial mmap failed: %m\n");
      return 1;
    }

  pthread_mutexattr_t ma;
  if (pthread_mutexattr_init (&ma) != 0)
    {
      puts ("mutexattr_init failed");
      return 1;
    }
  if (pthread_mutexattr_setpshared (&ma, 1) != 0)
    {
      puts ("mutexattr_setpshared failed");
      return 1;
    }
  if (pthread_mutex_init (&p->m, &ma) != 0)
    {
      puts ("mutex_init failed");
      return 1;
    }
  if (pthread_mutexattr_destroy (&ma) != 0)
    {
      puts ("mutexattr_destroy failed");
      return 1;
    }

  pthread_condattr_t ca;
  if (pthread_condattr_init (&ca) != 0)
    {
      puts ("condattr_init failed");
      return 1;
    }
  if (pthread_condattr_setpshared (&ca, 1) != 0)
    {
      puts ("condattr_setpshared failed");
      return 1;
    }
  if (pthread_cond_init (&p->c, &ca) != 0)
    {
      puts ("mutex_init failed");
      return 1;
    }
  if (pthread_condattr_destroy (&ca) != 0)
    {
      puts ("condattr_destroy failed");
      return 1;
    }

  if (pthread_mutex_lock (&p->m) != 0)
    {
      puts ("initial mutex_lock failed");
      return 1;
    }

  p->var = 42;

  pid_t pid = fork ();
  if (pid == -1)
    {
      printf ("fork failed: %m\n");
      return 1;
    }

  if (pid == 0)
    {
      void *oldp = p;
      p = mmap (NULL, sizeof (*p), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);

      if (p == oldp)
	{
	  puts ("child: mapped to same address");
	  kill (getppid (), SIGKILL);
	  exit (1);
	}

      munmap (oldp, sizeof (*p));

      if (pthread_mutex_lock (&p->m) != 0)
	{
	  puts ("child: mutex_lock failed");
	  kill (getppid (), SIGKILL);
	  exit (1);
	}

      p->var = 0;

#ifndef USE_COND_SIGNAL
      if (pthread_cond_broadcast (&p->c) != 0)
	{
	  puts ("child: cond_broadcast failed");
	  kill (getppid (), SIGKILL);
	  exit (1);
	}
#else
      if (pthread_cond_signal (&p->c) != 0)
	{
	  puts ("child: cond_signal failed");
	  kill (getppid (), SIGKILL);
	  exit (1);
	}
#endif

      if (pthread_mutex_unlock (&p->m) != 0)
	{
	  puts ("child: mutex_unlock failed");
	  kill (getppid (), SIGKILL);
	  exit (1);
	}

      exit (0);
    }

  do
    pthread_cond_wait (&p->c, &p->m);
  while (p->var != 0);

  if (TEMP_FAILURE_RETRY (waitpid (pid, NULL, 0)) != pid)
    {
      printf ("waitpid failed: %m\n");
      kill (pid, SIGKILL);
      return 1;
    }

  return 0;
}
