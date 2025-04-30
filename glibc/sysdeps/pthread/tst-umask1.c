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

#include <fcntl.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>


static struct
{
  int (*fp) (const char *, mode_t);
  const char *name;
  bool is_fd;
} fcts[] =
{
  { creat, "creat", true },
  { mkdir, "mkdir", false },
  { mkfifo, "mkfifo", false },
};
#define nfcts (sizeof (fcts) / sizeof (fcts[0]))


static int
work (const char *fname, int mask)
{
  int result = 0;
  size_t i;
  for (i = 0; i < nfcts; ++i)
    {
      remove (fname);
      int fd = fcts[i].fp (fname, 0777);
      if (fd == -1)
	{
	  printf ("cannot %s %s: %m\n", fcts[i].name, fname);
	  exit (1);
	}
      if (fcts[i].is_fd)
	close (fd);
      struct stat64 st;
      if (stat64 (fname, &st) == -1)
	{
	  printf ("cannot stat %s after %s: %m\n", fname, fcts[i].name);
	  exit (1);
	}

      if ((st.st_mode & mask) != 0)
	{
	  printf ("mask not successful after %s: %x still set\n",
		  fcts[i].name, (unsigned int) (st.st_mode & mask));
	  result = 1;
	}
    }

  return result;
}


static pthread_barrier_t bar;


static void *
tf (void *arg)
{
  pthread_barrier_wait (&bar);

  int result = work (arg, 022);

  pthread_barrier_wait (&bar);

  pthread_barrier_wait (&bar);

  return (work (arg, 0) | result) ? (void *) -1l : NULL;
}


static int
do_test (const char *fname)
{
  int result = 0;

  umask (0);
  result |= work (fname, 0);

  pthread_barrier_init (&bar, NULL, 2);

  pthread_t th;
  if (pthread_create (&th, NULL, tf, (void *) fname) != 0)
    {
      puts ("cannot create thread");
      exit (1);
    }

  umask (022);
  result |= work (fname, 022);

  pthread_barrier_wait (&bar);

  pthread_barrier_wait (&bar);

  umask (0);

  pthread_barrier_wait (&bar);

  void *res;
  if (pthread_join (th, &res) != 0)
    {
      puts ("join failed");
      exit (1);
    }

  remove (fname);

  return result || res != NULL;
}

#define TEST_FUNCTION do_test (argc < 2 ? "/tmp/tst-umask.tmp" : argv[1])
#include "../test-skeleton.c"
