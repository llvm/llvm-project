/* Test for shm_open cancellation handling: BZ #18243.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <pthread.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>

static sem_t sem;	/* Use to sync with thread start.  */
static char shm_name[sizeof "/glibc-shm_open-cancel-" + sizeof (pid_t) * 3];

static void
init_shm_name (void)
{
  snprintf (shm_name, sizeof (shm_name), "/glibc-shm_open-cancel-%u",
	    getpid ());
}

static void
remove_shm (int status, void *arg)
{
  shm_unlink (shm_name);
}

static void *
tf (void *arg)
{
  pthread_setcancelstate (PTHREAD_CANCEL_DISABLE, 0);

  if (sem_wait (&sem) != 0)
    {
      printf ("error: sem_wait failed: %m");
      exit (1);
    }

  if (pthread_setcancelstate (PTHREAD_CANCEL_ENABLE, 0) != 0)
    {
      printf ("error: pthread_setcancelstate failed: %m");
      exit (1);
    }

  /* Neither sem_unlink or sem_open should act on thread cancellation.  */
  shm_unlink (shm_name);
  on_exit (remove_shm, NULL);

  int fd = shm_open (shm_name, O_CREAT, 0600);
  if (fd == -1)
    {
      int exit_code;
      if (errno == ENOSYS || errno == EACCES)
	exit_code = 77;
      else
	exit_code = 1;
      exit (exit_code);
    }

  if (pthread_setcancelstate (PTHREAD_CANCEL_DISABLE, 0) != 0)
    {
      printf ("error: pthread_setcancelstate failed: %m");
      exit (1);
    }

  if (close (fd) != 0)
    {
      printf ("error: pthread_setcancelstate failed: %m");
      exit (1);
    }

  return NULL;
}

static int
do_test (void)
{
  pthread_t td;

  init_shm_name ();

  if (sem_init (&sem, 0, 0))
    {
      printf ("error: sem_init failed: %m\n");
      exit (1);
    }

  if (pthread_create (&td, NULL, tf, NULL) != 0)
    {
      printf ("error: pthread_create failed: %m\n");
      exit (1);
    }

  if (pthread_cancel (td) != 0)
    {
      printf ("error: pthread_cancel failed: %m\n");
      exit (1);
    }

  if (sem_post (&sem) != 0)
    {
      printf ("error: sem_post failed: %m\n");
      exit (1);
    }

  void *r;
  if (pthread_join (td, &r) != 0)
    {
      printf ("error: pthread_join failed: %m\n");
      exit (1);
    }

  if (r == PTHREAD_CANCELED)
    {
      puts ("error: pthread_join returned PTHREAD_CANCELED");
      exit (1);
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include <test-skeleton.c>
