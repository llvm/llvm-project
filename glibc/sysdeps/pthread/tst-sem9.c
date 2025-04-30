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

#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static void
remove_sem (int status, void *arg)
{
  sem_unlink (arg);
}


static int
do_test (void)
{
  sem_t *s;
  int i;

  on_exit (remove_sem, (void *) "/glibc-tst-sem9");

  for (i = 0; i < 3; ++i)
    {
      s = sem_open ("/glibc-tst-sem9", O_CREAT, 0600, 1);
      if (s == SEM_FAILED)
	{
	  if (errno == ENOSYS)
	    {
	      puts ("sem_open not supported.  Oh well.");
	      return 0;
	    }

	  /* Maybe the shm filesystem has strict permissions.  */
	  if (errno == EACCES)
	    {
	      puts ("sem_open not allowed.  Oh well.");
	      return 0;
	    }

	  printf ("sem_open: %m\n");
	  return 1;
	}

      /* Now close the handle.  */
      if (sem_close (s) != 0)
	{
	  puts ("sem_close failed");
	  return 1;
	}

      /* And remove it.  */
      if (sem_unlink ("/glibc-tst-sem9") != 0)
	{
	  puts ("sem_unlink failed");
	  return 1;
	}
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
