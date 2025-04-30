/* Test for SEM_VALUE_MAX overflow detection: BZ #18434.
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

#include <errno.h>
#include <limits.h>
#include <semaphore.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>


static int
do_test (void)
{
  sem_t s;

  if (sem_init (&s, 0, SEM_VALUE_MAX))
    {
      printf ("sem_init: %m\n");
      return 1;
    }

  int result = 0;

  int value = 0xdeadbeef;
  if (sem_getvalue (&s, &value))
    {
      printf ("sem_getvalue: %m\n");
      result = 1;
    }
  else
    {
      printf ("sem_getvalue after init: %d\n", value);
      if (value != SEM_VALUE_MAX)
	{
	  printf ("\tshould be %d\n", SEM_VALUE_MAX);
	  result = 1;
	}
    }

  errno = 0;
  if (sem_post(&s) == 0)
    {
      puts ("sem_post at SEM_VALUE_MAX succeeded!");
      result = 1;
    }
  else
    {
      printf ("sem_post at SEM_VALUE_MAX: %m (%d)\n", errno);
      if (errno != EOVERFLOW)
	{
	  printf ("\tshould be %s (EOVERFLOW = %d)\n",
		  strerror (EOVERFLOW), EOVERFLOW);
	  result = 1;
	}
    }

  value = 0xbad1d00d;
  if (sem_getvalue (&s, &value))
    {
      printf ("sem_getvalue: %m\n");
      result = 1;
    }
  else
    {
      printf ("sem_getvalue after post: %d\n", value);
      if (value != SEM_VALUE_MAX)
	{
	  printf ("\tshould be %d\n", SEM_VALUE_MAX);
	  result = 1;
	}
    }

  if (sem_destroy (&s))
    {
      printf ("sem_destroy: %m\n");
      result = 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
