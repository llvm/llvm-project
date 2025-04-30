/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>


typedef struct
  {
    pthread_cond_t cond;
    pthread_mutex_t lock;
    pthread_t h;
  } T;


static volatile bool done;


static void *
tf (void *arg)
{
  puts ("child created");

  if (pthread_setcancelstate (PTHREAD_CANCEL_ENABLE, NULL) != 0
      || pthread_setcanceltype (PTHREAD_CANCEL_DEFERRED, NULL) != 0)
    {
      puts ("cannot set cancellation options");
      exit (1);
    }

  T *t = (T *) arg;

  if (pthread_mutex_lock (&t->lock) != 0)
    {
      puts ("child: lock failed");
      exit (1);
    }

  done = true;

  if (pthread_cond_signal (&t->cond) != 0)
    {
      puts ("child: cond_signal failed");
      exit (1);
    }

  if (pthread_cond_wait (&t->cond, &t->lock) != 0)
    {
      puts ("child: cond_wait failed");
      exit (1);
    }

  if (pthread_mutex_unlock (&t->lock) != 0)
    {
      puts ("child: unlock failed");
      exit (1);
    }

  return NULL;
}


static int
do_test (void)
{
  int i;
#define N 100
  T *t[N];
  for (i = 0; i < N; ++i)
    {
      printf ("round %d\n", i);

      t[i] = (T *) malloc (sizeof (T));
      if (t[i] == NULL)
	{
	  puts ("out of memory");
	  exit (1);
	}

      if (pthread_mutex_init (&t[i]->lock, NULL) != 0
	  || pthread_cond_init (&t[i]->cond, NULL) != 0)
	{
	  puts ("an _init function failed");
	  exit (1);
	}

      if (pthread_mutex_lock (&t[i]->lock) != 0)
	{
	  puts ("initial mutex_lock failed");
	  exit (1);
	}

      done = false;

      if (pthread_create (&t[i]->h, NULL, tf, t[i]) != 0)
	{
	  puts ("pthread_create failed");
	  exit (1);
	}

      do
	if (pthread_cond_wait (&t[i]->cond, &t[i]->lock) != 0)
	  {
	    puts ("cond_wait failed");
	    exit (1);
	  }
      while (! done);

      /* Release the lock since the cancel handler will get it.  */
      if (pthread_mutex_unlock (&t[i]->lock) != 0)
	{
	  puts ("mutex_unlock failed");
	  exit (1);
	}

      if (pthread_cancel (t[i]->h) != 0)
	{
	  puts ("cancel failed");
	  exit (1);
	}

      puts ("parent: joining now");

      void *result;
      if (pthread_join (t[i]->h, &result) != 0)
	{
	  puts ("join failed");
	  exit (1);
	}

      if (result != PTHREAD_CANCELED)
	{
	  puts ("result != PTHREAD_CANCELED");
	  exit (1);
	}
    }

  for (i = 0; i < N; ++i)
    free (t[i]);

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
