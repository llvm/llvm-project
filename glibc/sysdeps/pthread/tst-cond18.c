/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2004.

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
#include <limits.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
bool exiting;
int fd, spins, nn;
enum { count = 8 };		/* Number of worker threads.  */

void *
tf (void *id)
{
  pthread_mutex_lock (&lock);

  if ((long) id == 0)
    {
      while (!exiting)
	{
	  if ((spins++ % 1000) == 0)
	    write (fd, ".", 1);
	  pthread_mutex_unlock (&lock);

	  pthread_mutex_lock (&lock);
	  int njobs = rand () % (count + 1);
	  nn = njobs;
	  if ((rand () % 30) == 0)
	    pthread_cond_broadcast (&cv);
	  else
	    while (njobs--)
	      pthread_cond_signal (&cv);
	}

      pthread_cond_broadcast (&cv);
    }
  else
    {
      while (!exiting)
	{
	  while (!nn && !exiting)
	    pthread_cond_wait (&cv, &lock);
	  --nn;
	  pthread_mutex_unlock (&lock);

	  pthread_mutex_lock (&lock);
	}
    }

  pthread_mutex_unlock (&lock);
  return NULL;
}

int
do_test (void)
{
  fd = open ("/dev/null", O_WRONLY);
  if (fd < 0)
    {
      printf ("couldn't open /dev/null, %m\n");
      return 1;
    }

  pthread_t th[count + 1];
  pthread_attr_t attr;
  int i, ret, sz;
  pthread_attr_init (&attr);
  sz = sysconf (_SC_PAGESIZE);
#ifdef PTHREAD_STACK_MIN
  if (sz < PTHREAD_STACK_MIN)
	  sz = PTHREAD_STACK_MIN;
#endif
  pthread_attr_setstacksize (&attr, sz);

  for (i = 0; i <= count; ++i)
    if ((ret = pthread_create (&th[i], &attr, tf, (void *) (long) i)) != 0)
      {
	errno = ret;
	printf ("pthread_create %d failed: %m\n", i);
	return 1;
      }

  struct timespec ts = { .tv_sec = 20, .tv_nsec = 0 };
  while (nanosleep (&ts, &ts) != 0);

  pthread_mutex_lock (&lock);
  exiting = true;
  pthread_mutex_unlock (&lock);

  for (i = 0; i < count; ++i)
    pthread_join (th[i], NULL);

  close (fd);
  return 0;
}

#define TEST_FUNCTION do_test ()
#define TIMEOUT 40
#include "../test-skeleton.c"
