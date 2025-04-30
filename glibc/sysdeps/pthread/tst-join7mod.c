/* Verify that TLS access in separate thread in a dlopened library does not
   deadlock - the module.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <atomic.h>

static pthread_t th;
static int running = 1;

static void *
test_run (void *p)
{
  while (atomic_load_relaxed (&running))
    printf ("Test running\n");
  printf ("Test finished\n");
  return NULL;
}

static void __attribute__ ((constructor))
do_init (void)
{
  int ret = pthread_create (&th, NULL, test_run, NULL);

  if (ret != 0)
    {
      printf ("failed to create thread: %s (%d)\n", strerror (ret), ret);
      exit (1);
    }
}

static void __attribute__ ((destructor))
do_end (void)
{
  atomic_store_relaxed (&running, 0);
  int ret = pthread_join (th, NULL);

  if (ret != 0)
    {
      printf ("pthread_join: %s(%d)\n", strerror (ret), ret);
      exit (1);
    }

  printf ("Thread joined\n");
}
