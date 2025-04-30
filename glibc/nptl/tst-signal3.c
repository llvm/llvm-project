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
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


#ifdef SIGRTMIN

/* Number of different signals to use.  Also is the number of threads.  */
# define N 10
/* Maximum number of threads in flight at any one time.  */
# define INFLIGHT 5
/* Number of signals sent in total.  */
# define ROUNDS 10000


static int received[N][N];
static int nsig[N];
static pthread_t th[N];
static sem_t sem;
static pthread_mutex_t lock[N];
static pthread_t th_main;
static int sig0;

static void
handler (int sig)
{
  int i;
  for (i = 0; i < N; ++i)
    if (pthread_equal (pthread_self (), th[i]))
      break;

  if (i == N)
    {
      if (pthread_equal (pthread_self (), th_main))
	puts ("signal received by main thread");
      else
	printf ("signal received by unknown thread (%lx)\n",
		(unsigned long int) pthread_self ());
      exit (1);
    }

  ++received[i][sig - sig0];

  sem_post (&sem);
}


static void *
tf (void *arg)
{
  int idx = (long int) arg;

  sigset_t ss;
  sigemptyset (&ss);

  int i;
  for (i = 0; i <= idx; ++i)
    sigaddset (&ss, sig0 + i);

  if (pthread_sigmask (SIG_UNBLOCK, &ss, NULL) != 0)
    {
      printf ("thread %d: pthread_sigmask failed\n", i);
      exit (1);
    }

  pthread_mutex_lock (&lock[idx]);

  return NULL;
}


static int
do_test (void)
{
  /* Block all signals.  */
  sigset_t ss;
  sigfillset (&ss);

  th_main = pthread_self ();

  sig0 = SIGRTMIN;

  if (pthread_sigmask (SIG_SETMASK, &ss, NULL) != 0)
    {
      puts ("1st pthread_sigmask failed");
      exit (1);
    }

  /* Install the handler.  */
  int i;
  for (i = 0; i < N; ++i)
    {
      struct sigaction sa =
	{
	  .sa_handler = handler,
	  .sa_flags = 0
	};
      sigfillset (&sa.sa_mask);

      if (sigaction (sig0 + i, &sa, NULL) != 0)
	{
	  printf ("sigaction for signal %d failed\n", i);
	  exit (1);
	}
    }

  if (sem_init (&sem, 0, INFLIGHT) != 0)
    {
      puts ("sem_init failed");
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

  for (i = 0; i < N; ++i)
    {
      if (pthread_mutex_init (&lock[i], NULL) != 0)
	{
	  printf ("mutex_init[%d] failed\n", i);
	}

      if (pthread_mutex_lock (&lock[i]) != 0)
	{
	  printf ("mutex_lock[%d] failed\n", i);
	}

      if (pthread_create (&th[i], &a, tf, (void *) (long int) i) != 0)
	{
	  printf ("create of thread %d failed\n", i);
	  exit (1);
	}
    }

  if (pthread_attr_destroy (&a) != 0)
    {
      puts ("attr_destroy failed");
      exit (1);
    }

  int result = 0;
  unsigned int r = 42;
  pid_t pid = getpid ();

  for (i = 0; i < ROUNDS; ++i)
    {
      if (TEMP_FAILURE_RETRY (sem_wait (&sem)) != 0)
	{
	  printf ("sem_wait round %d failed: %m\n", i);
	  exit (1);
	}

      int s = rand_r (&r) % N;

      kill (pid, sig0 + s);
    }

  void *status;
  for (i = 0; i < N; ++i)
    {
      if (pthread_mutex_unlock (&lock[i]) != 0)
	{
	  printf ("unlock %d failed\n", i);
	  exit (1);
	}

      if (pthread_join (th[i], &status) != 0)
	{
	  printf ("join %d failed\n", i);
	  result = 1;
	}
      else if (status != NULL)
	{
	  printf ("%d: result != NULL\n", i);
	  result = 1;
	}
    }

  int total = 0;
  for (i = 0; i < N; ++i)
    {
      int j;

      for (j = 0; j <= i; ++j)
	total += received[i][j];

      for (j = i + 1; j < N; ++j)
	if (received[i][j] != 0)
	  {
	    printf ("thread %d received signal SIGRTMIN+%d\n", i, j);
	    result = 1;
	  }
    }

  if (total != ROUNDS)
    {
      printf ("total number of handled signals is %d, expected %d\n",
	      total, ROUNDS);
      result = 1;
    }

  printf ("A total of %d signals sent and received\n", total);
  for (i = 0; i < N; ++i)
    {
      printf ("thread %2d:", i);

      int j;
      for (j = 0; j <= i; ++j)
	{
	  printf (" %5d", received[i][j]);
	  nsig[j] += received[i][j];
	}

      putchar ('\n');

    }

  printf ("\nTotal    :");
  for (i = 0; i < N; ++i)
    printf (" %5d", nsig[i]);
  putchar ('\n');

  return result;
}

# define TEST_FUNCTION do_test ()

#else
# define TEST_FUNCTION 0
#endif

#include "../test-skeleton.c"
