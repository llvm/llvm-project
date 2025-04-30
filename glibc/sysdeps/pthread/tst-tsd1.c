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

#include <pthread.h>
#include <stdio.h>
#include <string.h>


static int
do_test (void)
{
  pthread_key_t key1;
  pthread_key_t key2;
  void *value;
  /* Addresses of val1 and val2 are used as arbitrary but valid pointers
     in calls to pthread_setspecific to avoid GCC warnings.  */
  char val1 = 0, val2 = 0;
  int result = 0;
  int err;

  err = pthread_key_create (&key1, NULL);
  if (err != 0)
    {
      printf ("1st key_create failed: %s\n", strerror (err));
      return 1;
    }

  /* Initial value must be NULL.  */
  value = pthread_getspecific (key1);
  if (value != NULL)
    {
      puts ("1st getspecific != NULL");
      result = 1;
    }

  err = pthread_setspecific (key1, (void *) &val1);
  if (err != 0)
    {
      printf ("1st setspecific failed: %s\n", strerror (err));
      return 1;
    }

  value = pthread_getspecific (key1);
  if (value == NULL)
    {
      puts ("2nd getspecific == NULL\n");
      result = 1;
    }
  else if (value != (void *) &val1)
    {
      puts ("2nd getspecific != &val1l\n");
      result = 1;
    }

  err = pthread_setspecific (key1, (void *) &val2);
  if (err != 0)
    {
      printf ("2nd setspecific failed: %s\n", strerror (err));
      return 1;
    }

  value = pthread_getspecific (key1);
  if (value == NULL)
    {
      puts ("3rd getspecific == NULL\n");
      result = 1;
    }
  else if (value != (void *) &val2)
    {
      puts ("3rd getspecific != &val2\n");
      result = 1;
    }

  err = pthread_key_delete (key1);
  if (err != 0)
    {
      printf ("key_delete failed: %s\n", strerror (err));
      result = 1;
    }


  err = pthread_key_create (&key2, NULL);
  if (err != 0)
    {
      printf ("2nd key_create failed: %s\n", strerror (err));
      return 1;
    }

  if (key1 != key2)
    puts ("key1 != key2; no more tests performed");
  else
    {
      value = pthread_getspecific (key2);
      if (value != NULL)
	{
	  puts ("4th getspecific != NULL");
	  result = 1;
	}
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
