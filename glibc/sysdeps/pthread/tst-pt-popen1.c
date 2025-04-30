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
#include <error.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void *
dummy (void *x)
{
  return NULL;
}

static char buf[sizeof "something\n"];

static int
do_test (void)
{
  FILE *f;
  pthread_t p;
  int err;

  f = popen ("echo something", "r");
  if (f == NULL)
    error (EXIT_FAILURE, errno, "popen failed");
  if (fgets (buf, sizeof (buf), f) == NULL)
    error (EXIT_FAILURE, 0, "fgets failed");
  if (strcmp (buf, "something\n"))
    error (EXIT_FAILURE, 0, "read wrong data");
  if (pclose (f))
    error (EXIT_FAILURE, errno, "pclose returned non-zero");
  if ((err = pthread_create (&p, NULL, dummy, NULL)))
    error (EXIT_FAILURE, err, "pthread_create failed");
  if ((err = pthread_join (p, NULL)))
    error (EXIT_FAILURE, err, "pthread_join failed");
  exit (0);
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
