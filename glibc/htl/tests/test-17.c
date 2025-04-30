/* Test that the key reuse inside libpthread does not cause thread
   specific values to persist.
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

#define _GNU_SOURCE 1

#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include <errno.h>

void
work (int iter)
{
  error_t err;
  pthread_key_t key1;
  pthread_key_t key2;
  void *value1;
  void *value2;

  printf ("work/%d: start\n", iter);
  err = pthread_key_create (&key1, NULL);
  assert (err == 0);
  err = pthread_key_create (&key2, NULL);
  assert (err == 0);

  value1 = pthread_getspecific (key1);
  value2 = pthread_getspecific (key2);
  printf ("work/%d: pre-setspecific: %p,%p\n", iter, value1, value2);
  assert (value1 == NULL);
  assert (value2 == NULL);
  err = pthread_setspecific (key1, (void *) (0x100 + iter));
  assert (err == 0);
  err = pthread_setspecific (key2, (void *) (0x200 + iter));
  assert (err == 0);

  value1 = pthread_getspecific (key1);
  value2 = pthread_getspecific (key2);
  printf ("work/%d: post-setspecific: %p,%p\n", iter, value1, value2);
  assert (value1 == (void *) (0x100 + iter));
  assert (value2 == (void *) (0x200 + iter));

  err = pthread_key_delete (key1);
  assert (err == 0);
  err = pthread_key_delete (key2);
  assert (err == 0);
}

int
main (int argc, char *argv[])
{
  int i;

  for (i = 0; i < 8; ++i)
    work (i + 1);

  return 0;
}
