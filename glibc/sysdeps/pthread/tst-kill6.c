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
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static int do_test (void);

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

static pthread_t receiver;
static sem_t sem;
static pthread_barrier_t b;

static void
handler (int sig)
{
  if (sig != SIGUSR1)
    {
      write_message ("wrong signal\n");
      _exit (1);
    }

  if (pthread_self () != receiver)
    {
      write_message ("not the intended receiver\n");
      _exit (1);
    }

  if (sem_post (&sem) != 0)
    {
      write_message ("sem_post failed\n");
      _exit (1);
    }
}


static void *
tf (void *a)
{
  int e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("child: barrier_wait failed");
      exit (1);
    }

  return NULL;
}


int
do_test (void)
{
  struct sigaction sa;
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = 0;
  sa.sa_handler = handler;
  if (sigaction (SIGUSR1, &sa, NULL) != 0)
    {
      puts ("sigaction failed");
      exit (1);
    }

#define N 20

  if (pthread_barrier_init (&b, NULL, N + 1) != 0)
    {
      puts ("barrier_init failed");
      exit (1);
    }

  pthread_attr_t a;

  if (pthread_attr_init (&a) != 0)
    {
      puts ("attr_init failed");
      exit (1);
    }

  if (pthread_attr_setstacksize (&a, 1 * 1024 * 1024) != 0)
    {
      puts ("attr_setstacksize failed");
      return 1;
    }

  pthread_t th[N];
  int i;
  for (i = 0; i < N; ++i)
    if (pthread_create (&th[i], &a, tf, NULL) != 0)
      {
	puts ("create failed");
	exit (1);
      }

  if (pthread_attr_destroy (&a) != 0)
    {
      puts ("attr_destroy failed");
      exit (1);
    }

  if (sem_init (&sem, 0, 0) != 0)
    {
      puts ("sem_init failed");
      exit (1);
    }

  for (i = 0; i < N * 10; ++i)
    {
      receiver = th[i % N];

      if (pthread_kill (receiver, SIGUSR1) != 0)
	{
	  puts ("kill failed");
	  exit (1);
	}

      if (TEMP_FAILURE_RETRY (sem_wait (&sem)) != 0)
	{
	  puts ("sem_wait failed");
	  exit (1);
	}
    }

  int e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  for (i = 0; i < N; ++i)
    if (pthread_join (th[i], NULL) != 0)
      {
	puts ("join failed");
	exit (1);
      }

  return 0;
}
