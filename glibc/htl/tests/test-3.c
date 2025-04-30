/* Test the thread attribute get and set methods.
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
#include <sched.h>
#include <assert.h>
#include <errno.h>

int
main (int argc, char *argv[])
{
  error_t err;
  pthread_attr_t attr;

  int i;
  struct sched_param sp;
  void *p;
  size_t sz;

  err = pthread_attr_init (&attr);
  assert_perror (err);

  err = pthread_attr_destroy (&attr);
  assert_perror (err);

  err = pthread_attr_init (&attr);
  assert_perror (err);

#define TEST1(foo, rv, v) \
	err = pthread_attr_get##foo (&attr, rv); \
	assert_perror (err); \
	\
	err = pthread_attr_set##foo (&attr, v); \
	assert_perror (err);

#define TEST(foo, rv, v) TEST1(foo, rv, v)

  TEST (inheritsched, &i, i);
  TEST (schedparam, &sp, &sp);
  TEST (schedpolicy, &i, i);
  TEST (scope, &i, i);
  TEST (stackaddr, &p, p);
  TEST (detachstate, &i, i);
  TEST (guardsize, &sz, sz);
  TEST (stacksize, &sz, sz);

  err = pthread_attr_getstack (&attr, &p, &sz);
  assert_perror (err);

  err = pthread_attr_setstack (&attr, p, sz);
  assert_perror (err);

  return 0;
}
