/* Test pthread_once.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#define _GNU_SOURCE

#include <pthread.h>
#include <assert.h>
#include <error.h>
#include <errno.h>

#define THREADS 10

pthread_once_t inc_var_once = PTHREAD_ONCE_INIT;
int var;

void
inc_var (void)
{
  var++;
}

void *
thr (void *arg)
{
  int i;

  for (i = 0; i < 500; i++)
    pthread_once (&inc_var_once, inc_var);

  return 0;
}

int
main (int argc, char **argv)
{
  error_t err;
  int i;
  pthread_t tid[THREADS];

  for (i = 0; i < THREADS; i++)
    {
      err = pthread_create (&tid[i], 0, thr, 0);
      if (err)
	error (1, err, "pthread_create (%d)", i);
    }

  assert (thr (0) == 0);

  for (i = 0; i < THREADS; i++)
    {
      void *ret;

      err = pthread_join (tid[i], &ret);
      if (err)
	error (1, err, "pthread_join");

      assert (ret == 0);
    }

  assert (var == 1);

  return 0;
}
