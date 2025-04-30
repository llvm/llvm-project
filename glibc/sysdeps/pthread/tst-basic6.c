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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


static char *p;

static pthread_barrier_t b;
#define BT \
  e = pthread_barrier_wait (&b);					      \
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)			      \
    {									      \
      puts ("barrier_wait failed");					      \
      exit (1);								      \
    }


static void *
tf (void *a)
{
  int e;

  BT;

  char *p2 = getcwd (NULL, 0);
  if (p2 == NULL)
    {
      puts ("2nd getcwd failed");
      exit (1);
    }

  if (strcmp (p, p2) != 0)
    {
      printf ("initial cwd mismatch: \"%s\" vs \"%s\"\n", p, p2);
      exit (1);
    }

  free (p);
  free (p2);

  if (chdir ("..") != 0)
    {
      puts ("chdir failed");
      exit (1);
    }

  p = getcwd (NULL, 0);
  if (p == NULL)
    {
      puts ("getcwd failed");
      exit (1);
    }

  return a;
}


int
do_test (void)
{
  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      exit (1);
    }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      exit (1);
    }

  p = getcwd (NULL, 0);
  if (p == NULL)
    {
      puts ("getcwd failed");
      exit (1);
    }

  int e;
  BT;

  if (pthread_join (th, NULL) != 0)
    {
      puts ("join failed");
      exit (1);
    }

  char *p2 = getcwd (NULL, 0);
  if (p2 == NULL)
    {
      puts ("2nd getcwd failed");
      exit (1);
    }

  if (strcmp (p, p2) != 0)
    {
      printf ("cwd after chdir mismatch: \"%s\" vs \"%s\"\n", p, p2);
      exit (1);
    }

  free (p);
  free (p2);

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
