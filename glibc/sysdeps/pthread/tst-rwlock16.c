/* Copyright (C) 2015-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

/* This tests that with a reader-preferring rwlock, all readers are woken if
   one reader "steals" lock ownership from a blocked writer.  */

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>
#include <unistd.h>

/* If we strictly prefer writers over readers, a program must not expect
   that, in the presence of concurrent writers, one reader will also acquire
   the lock when another reader has already done so.  Thus, use the
   default rwlock type that does not strictly prefer writers.  */
static pthread_rwlock_t r = PTHREAD_RWLOCK_INITIALIZER;

static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;

/* Avoid using glibc-internal atomic operations.  */
static sem_t stop;
static int consumer_stop = 0;

static void *
writer (void *arg)
{
  int s;

  do
    {
      if (pthread_rwlock_wrlock (&r) != 0)
	{
	  puts ("wrlock failed");
	  exit (EXIT_FAILURE);
	}
      if (pthread_rwlock_unlock (&r) != 0)
	{
	  puts ("unlock failed");
	  exit (EXIT_FAILURE);
	}
      sem_getvalue (&stop, &s);
    }
  while (s == 0);
  return NULL;
}

static void *
reader_producer (void *arg)
{
  int s;

  do
    {
      if (pthread_rwlock_rdlock (&r) != 0)
	{
	  puts ("rdlock reader failed");
	  exit (EXIT_FAILURE);
	}

      sem_getvalue (&stop, &s);

      pthread_mutex_lock (&m);
      if (s != 0)
	consumer_stop = 1;
      pthread_cond_signal (&cv);
      pthread_mutex_unlock (&m);

      if (pthread_rwlock_unlock (&r) != 0)
	{
	  puts ("unlock reader failed");
	  exit (EXIT_FAILURE);
	}
    }
  while (s == 0);
  puts ("producer finished");
  return NULL;
}

static void *
reader_consumer (void *arg)
{
  int s;

  do
    {
      if (pthread_rwlock_rdlock (&r) != 0)
	{
	  puts ("rdlock reader failed");
	  exit (EXIT_FAILURE);
	}

      pthread_mutex_lock (&m);
      s = consumer_stop;
      if (s == 0)
	pthread_cond_wait (&cv, &m);
      pthread_mutex_unlock (&m);

      if (pthread_rwlock_unlock (&r) != 0)
	{
	  puts ("unlock reader failed");
	  exit (EXIT_FAILURE);
	}
    }
  while (s == 0);
    puts ("consumer finished");
  return NULL;
}


static int
do_test (void)
{
  pthread_t w1, w2, rp, rc;

  if (pthread_create (&w1, NULL, writer, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }
  if (pthread_create (&w2, NULL, writer, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }
  if (pthread_create (&rc, NULL, reader_consumer, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }
  if (pthread_create (&rp, NULL, reader_producer, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }

  sleep (2);
  sem_post (&stop);

  if (pthread_join (w1, NULL) != 0)
    {
      puts ("w1 join failed");
      return 1;
    }
  if (pthread_join (w2, NULL) != 0)
    {
      puts ("w2 join failed");
      return 1;
    }
  if (pthread_join (rp, NULL) != 0)
    {
      puts ("reader_producer join failed");
      return 1;
    }
  if (pthread_join (rc, NULL) != 0)
    {
      puts ("reader_consumer join failed");
      return 1;
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
