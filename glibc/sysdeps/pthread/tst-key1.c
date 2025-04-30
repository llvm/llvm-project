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

#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static int do_test (void);

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

int
do_test (void)
{
  int max;
#ifdef PTHREAD_KEYS_MAX
  max = PTHREAD_KEYS_MAX;
#else
  max = _POSIX_THREAD_KEYS_MAX;
#endif
  pthread_key_t *keys = alloca (max * sizeof (pthread_key_t));

  int i;
  for (i = 0; i < max; ++i)
    if (pthread_key_create (&keys[i], NULL) != 0)
      {
	write_message ("key_create failed\n");
	_exit (1);
      }
    else
      {
	printf ("created key %d\n", i);

	if (pthread_setspecific (keys[i], (const void *) (i + 100l)) != 0)
	  {
	    write (2, "setspecific failed\n", 19);
	    _exit (1);
	  }
      }

  for (i = 0; i < max; ++i)
    {
      if (pthread_getspecific (keys[i]) != (void *) (i + 100l))
	{
	  write (2, "getspecific failed\n", 19);
	  _exit (1);
	}

      if (pthread_key_delete (keys[i]) != 0)
	{
	  write (2, "key_delete failed\n", 18);
	  _exit (1);
	}
    }

  /* Now it must be once again possible to allocate keys.  */
  if (pthread_key_create (&keys[0], NULL) != 0)
    {
      write (2, "2nd key_create failed\n", 22);
      _exit (1);
    }

  if (pthread_key_delete (keys[0]) != 0)
    {
      write (2, "2nd key_delete failed\n", 22);
      _exit (1);
    }

  return 0;
}
