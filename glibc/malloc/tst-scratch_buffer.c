/*
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

#include <array_length.h>
#include <scratch_buffer.h>
#include <support/check.h>
#include <support/support.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool
unchanged_array_size (struct scratch_buffer *buf, size_t a, size_t b)
{
  size_t old_length = buf->length;
  if (!scratch_buffer_set_array_size (buf, a, b))
    {
      printf ("scratch_buffer_set_array_size failed: %zu %zu\n",
	      a, b);
      return false;
    }
  if (old_length != buf->length)
    {
      printf ("scratch_buffer_set_array_size did not preserve size: %zu %zu\n",
	      a, b);
      return false;
    }
  return true;
}

static bool
array_size_must_fail (size_t a, size_t b)
{
  for (int pass = 0; pass < 2; ++pass)
    {
      struct scratch_buffer buf;
      scratch_buffer_init (&buf);
      if (pass > 0)
	if (!scratch_buffer_grow (&buf))
	  {
	    printf ("scratch_buffer_grow in array_size_must_fail failed\n");
	    return false;
	  }
      if (scratch_buffer_set_array_size (&buf, a, b))
	{
	  printf ("scratch_buffer_set_array_size passed: %d %zu %zu\n",
		  pass, a, b);
	  return false;
	}
      if (buf.data != buf.__space.__c)
	{
	  printf ("scratch_buffer_set_array_size did not free: %d %zu %zu\n",
		  pass, a, b);
	  return false;
	}
    }
  return true;
}

static int
do_test (void)
{
  {
    struct scratch_buffer buf;
    scratch_buffer_init (&buf);
    memset (buf.data, ' ', buf.length);
    scratch_buffer_free (&buf);
  }
  {
    struct scratch_buffer buf;
    scratch_buffer_init (&buf);
    memset (buf.data, ' ', buf.length);
    size_t old_length = buf.length;
    scratch_buffer_grow (&buf);
    if (buf.length <= old_length)
      {
	printf ("scratch_buffer_grow did not enlarge buffer\n");
	return 1;
      }
    memset (buf.data, ' ', buf.length);
    scratch_buffer_free (&buf);
  }
  {
    struct scratch_buffer buf;
    scratch_buffer_init (&buf);
    memset (buf.data, '@', buf.length);
    strcpy (buf.data, "prefix");
    size_t old_length = buf.length;
    scratch_buffer_grow_preserve (&buf);
    if (buf.length <= old_length)
      {
	printf ("scratch_buffer_grow_preserve did not enlarge buffer\n");
	return 1;
      }
    if (strcmp (buf.data, "prefix") != 0)
      {
	printf ("scratch_buffer_grow_preserve did not copy buffer\n");
	return 1;
      }
    for (unsigned i = 7; i < old_length; ++i)
      if (((char *)buf.data)[i] != '@')
	{
	  printf ("scratch_buffer_grow_preserve did not copy buffer (%u)\n",
		  i);
	  return 1;
	}
    scratch_buffer_free (&buf);
  }
  {
    struct scratch_buffer buf;
    scratch_buffer_init (&buf);
    for (int pass = 0; pass < 4; ++pass)
      {
	if (!(unchanged_array_size (&buf, 0, 0)
	      && unchanged_array_size (&buf, 1, 0)
	      && unchanged_array_size (&buf, 0, 1)
	      && unchanged_array_size (&buf, -1, 0)
	      && unchanged_array_size (&buf, 0, -1)
	      && unchanged_array_size (&buf, 1ULL << 16, 0)
	      && unchanged_array_size (&buf, 0, 1ULL << 16)
	      && unchanged_array_size (&buf, (size_t) (1ULL << 32), 0)
	      && unchanged_array_size (&buf, 0, (size_t) (1ULL << 32))))
	  return 1;
	if (!scratch_buffer_grow (&buf))
	  {
	    printf ("scratch_buffer_grow_failed (pass %d)\n", pass);
	  }
      }
    scratch_buffer_free (&buf);
  }
  {
    if (!(array_size_must_fail (-1, 1)
	  && array_size_must_fail (-1, -1)
	  && array_size_must_fail (1, -1)
	  && array_size_must_fail (((size_t)-1) / 4, 4)
	  && array_size_must_fail (4, ((size_t)-1) / 4)))
	return 1;
  }
  {
    struct scratch_buffer buf;
    scratch_buffer_init (&buf);
    memset (buf.data, '@', buf.length);

    size_t sizes[] = { 16, buf.length, buf.length + 16 };
    for (int i = 0; i < array_length (sizes); i++)
      {
        /* The extra size is unitialized through realloc.  */
        size_t l = sizes[i] > buf.length ? sizes[i] : buf.length;
        void *r = scratch_buffer_dupfree (&buf, l);
        void *c = xmalloc (l);
        memset (c, '@', l);
        TEST_COMPARE_BLOB (r, l, buf.data, l);
        free (r);
        free (c);
      }

    scratch_buffer_free (&buf);
  }
  return 0;
}

#include <support/test-driver.c>
