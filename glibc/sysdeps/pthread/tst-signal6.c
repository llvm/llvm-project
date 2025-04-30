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

#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


#ifdef SIGRTMIN

# define N 2
static pthread_barrier_t bar;
static struct
{
  void *p;
  pthread_t s;
} ti[N];
static int sig1;


static void
handler (int sig)
{
  pthread_t self = pthread_self ();
  size_t i;

  for (i = 0; i < N; ++i)
    if (ti[i].s == self)
      {
	if ((uintptr_t) ti[i].p <= (uintptr_t) &self
	    && (uintptr_t) ti[i].p + 2 * MINSIGSTKSZ > (uintptr_t) &self)
	  {
	    puts ("alt stack not used");
	    exit (1);
	  }

	printf ("thread %zu used alt stack for signal %d\n", i, sig);

	return;
      }

  puts ("handler: thread not found");
  exit (1);
}


static void *
tf (void *arg)
{
  size_t nr = (uintptr_t) arg;
  if (nr >= N)
    {
      puts ("wrong nr parameter");
      exit (1);
    }

  sigset_t ss;
  sigemptyset (&ss);
  size_t i;
  for (i = 0; i < N; ++i)
    if (i != nr)
      if (sigaddset (&ss, sig1 + i) != 0)
	{
	  puts ("tf: sigaddset failed");
	  exit (1);
	}
  if (pthread_sigmask (SIG_BLOCK, &ss, NULL) != 0)
    {
      puts ("tf: sigmask failed");
      exit (1);
    }

  void *p = malloc (2 * MINSIGSTKSZ);
  if (p == NULL)
    {
      puts ("tf: malloc failed");
      exit (1);
    }

  stack_t s;
  s.ss_sp = p;
  s.ss_size = 2 * MINSIGSTKSZ;
  s.ss_flags = 0;
  if (sigaltstack (&s, NULL) != 0)
    {
      puts ("tf: sigaltstack failed");
      exit (1);
    }

  ti[nr].p = p;
  ti[nr].s = pthread_self ();

  pthread_barrier_wait (&bar);

  pthread_barrier_wait (&bar);

  return NULL;
}


static int
do_test (void)
{
  sig1 = SIGRTMIN;
  if (sig1 + N > SIGRTMAX)
    {
      puts ("too few RT signals");
      return 0;
    }

  struct sigaction sa;
  sa.sa_handler = handler;
  sa.sa_flags = 0;
  sigemptyset (&sa.sa_mask);

  if (sigaction (sig1, &sa, NULL) != 0
      || sigaction (sig1 + 1, &sa, NULL) != 0
      || sigaction (sig1 + 2, &sa, NULL) != 0)
    {
      puts ("sigaction failed");
      return 1;
    }

  if (pthread_barrier_init (&bar, NULL, 1 + N) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  pthread_t th[N];
  size_t i;
  for (i = 0; i < N; ++i)
    if (pthread_create (&th[i], NULL, tf, (void *) (long int) i) != 0)
      {
	puts ("create failed");
	return 1;
      }

  /* Block the three signals.  */
  sigset_t ss;
  sigemptyset (&ss);
  for (i = 0; i <= N; ++i)
    sigaddset (&ss, sig1 + i);
  if (pthread_sigmask (SIG_BLOCK, &ss, NULL) != 0)
    {
      puts ("main: sigmask failed");
      return 1;
    }

  pthread_barrier_wait (&bar);

  /* Send some signals.  */
  pid_t me = getpid ();
  kill (me, sig1 + N);
  for (i = 0; i < N; ++i)
    kill (me, sig1 + i);
  kill (me, sig1 + N);

  /* Give the signals a chance to be worked on.  */
  sleep (1);

  pthread_barrier_wait (&bar);

  for (i = 0; i < N; ++i)
    if (pthread_join (th[i], NULL) != 0)
      {
	puts ("join failed");
	return 1;
      }

  return 0;
}

# define TEST_FUNCTION do_test ()

#else
# define TEST_FUNCTION 0
#endif
#include "../test-skeleton.c"
