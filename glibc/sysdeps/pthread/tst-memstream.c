/* Test for open_memstream locking.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* This test checks if concurrent writes to a FILE created with
   open_memstream are correctly interleaved without loss or corruption
   of data.  Large number of concurrent fputc operations are used
   and in the end the bytes written to the memstream buffer are
   counted to see if they all got recorded.

   This is a regression test to see if the single threaded stdio
   optimization broke multi-threaded open_memstream usage.  */

#include <pthread.h>
#include <stdio.h>
#include <support/check.h>
#include <support/xthread.h>

enum
{
  thread_count = 2,
  byte_count = 1000000,
};

struct closure
{
  FILE *fp;
  char b;
};

static void *
thread_func (void *closure)
{
  struct closure *args = closure;

  for (int i = 0; i < byte_count; ++i)
    fputc (args->b, args->fp);

  return NULL;
}

int
do_test (void)
{
  char *buffer = NULL;
  size_t buffer_length = 0;
  FILE *fp = open_memstream (&buffer, &buffer_length);
  if (fp == NULL)
    FAIL_RET ("open_memstream: %m");

  pthread_t threads[thread_count];
  struct closure args[thread_count];

  for (int i = 0; i < thread_count; ++i)
    {
      args[i].fp = fp;
      args[i].b = 'A' + i;
      threads[i] = xpthread_create (NULL, thread_func, args + i);
    }

  for (int i = 0; i < thread_count; ++i)
    xpthread_join (threads[i]);

  fclose (fp);

  if (buffer_length != thread_count * byte_count)
    FAIL_RET ("unexpected number of written bytes: %zu (should be %d)",
	      buffer_length, thread_count * byte_count);

  /* Verify that each thread written its unique character byte_count times.  */
  size_t counts[thread_count] = { 0, };
  for (size_t i = 0; i < buffer_length; ++i)
    {
      if (buffer[i] < 'A' || buffer[i] >= 'A' + thread_count)
	FAIL_RET ("written byte at %zu out of range: %d", i, buffer[i] & 0xFF);
      ++counts[buffer[i] - 'A'];
    }
  for (int i = 0; i < thread_count; ++i)
    if (counts[i] != byte_count)
      FAIL_RET ("incorrect write count for thread %d: %zu (should be %d)", i,
		counts[i], byte_count);

  return 0;
}

#define TIMEOUT 100
#include <support/test-driver.c>
