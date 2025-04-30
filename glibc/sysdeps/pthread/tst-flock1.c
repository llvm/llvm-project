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

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/file.h>


static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static int fd;


static void *
tf (void *arg)
{
  if (flock (fd, LOCK_SH | LOCK_NB) != 0)
    {
      puts ("second flock failed");
      exit (1);
    }

  pthread_mutex_unlock (&lock);

  return NULL;
}


static int
do_test (void)
{
  char tmp[] = "/tmp/tst-flock1-XXXXXX";

  fd = mkstemp (tmp);
  if (fd == -1)
    {
      puts ("mkstemp failed");
      exit (1);
    }

  unlink (tmp);

  write (fd, "foobar xyzzy", 12);

  if (flock (fd, LOCK_EX | LOCK_NB) != 0)
    {
      puts ("first flock failed");
      exit (1);
    }

  pthread_mutex_lock (&lock);

  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("pthread_create failed");
      exit (1);
    }

  pthread_mutex_lock (&lock);

  void *result;
  if (pthread_join (th, &result) != 0)
    {
      puts ("pthread_join failed");
      exit (1);
    }

  close (fd);

  return result != NULL;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
