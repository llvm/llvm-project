/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2004.

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


static void *
tf (void *arg)
{
  int fds[2];
  if (pipe (fds) != 0)
    {
      puts ("pipe failed");
      exit (1);
    }

  char buf[10];
  read (fds[0], buf, sizeof (buf));

  puts ("read returned");
  exit (1);
}

static pthread_t th;

static void
__attribute ((destructor))
dest (void)
{
  if (pthread_cancel (th) != 0)
    {
      puts ("cancel failed");
      _exit (1);
    }
  void *r;
  if (pthread_join (th, &r) != 0)
    {
      puts ("join failed");
      _exit (1);
    }
  /* Exit successfully.  */
  _exit (0);
}

void
m (void)
{
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      _exit (1);
    }
}
