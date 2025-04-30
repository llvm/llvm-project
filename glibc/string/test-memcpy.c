/* Test and measure memcpy functions.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Jakub Jelinek <jakub@redhat.com>, 1999.

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

#ifndef MEMCPY_RESULT
# define MEMCPY_RESULT(dst, len) dst
# define MIN_PAGE_SIZE 131072
# define TEST_MAIN
# define TEST_NAME "memcpy"
# include "test-string.h"

char *simple_memcpy (char *, const char *, size_t);
char *builtin_memcpy (char *, const char *, size_t);

IMPL (simple_memcpy, 0)
IMPL (builtin_memcpy, 0)
IMPL (memcpy, 1)

char *
simple_memcpy (char *dst, const char *src, size_t n)
{
  char *ret = dst;
  while (n--)
    *dst++ = *src++;
  return ret;
}

char *
builtin_memcpy (char *dst, const char *src, size_t n)
{
  return __builtin_memcpy (dst, src, n);
}
#endif

typedef char *(*proto_t) (char *, const char *, size_t);

static void
do_one_test (impl_t *impl, char *dst, const char *src,
	     size_t len)
{
  size_t i;

  /* Must clear the destination buffer set by the previous run.  */
  for (i = 0; i < len; i++)
    dst[i] = 0;

  if (CALL (impl, dst, src, len) != MEMCPY_RESULT (dst, len))
    {
      error (0, 0, "Wrong result in function %s %p %p", impl->name,
	     CALL (impl, dst, src, len), MEMCPY_RESULT (dst, len));
      ret = 1;
      return;
    }

  if (memcmp (dst, src, len) != 0)
    {
      error (0, 0, "Wrong result in function %s dst %p \"%.*s\" src %p \"%.*s\" len %zu",
	     impl->name, dst, (int) len, dst, src, (int) len, src, len);
      ret = 1;
      return;
    }
}

static void
do_test (size_t align1, size_t align2, size_t len)
{
  size_t i, j;
  char *s1, *s2;

  align1 &= 4095;
  if (align1 + len >= page_size)
    return;

  align2 &= 4095;
  if (align2 + len >= page_size)
    return;

  s1 = (char *) (buf1 + align1);
  s2 = (char *) (buf2 + align2);

  for (i = 0, j = 1; i < len; i++, j += 23)
    s1[i] = j;

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, s2, s1, len);
}

static void
do_random_tests (void)
{
  size_t i, j, n, align1, align2, len, size1, size2, size;
  int c;
  unsigned char *p1, *p2;
  unsigned char *res;

  for (n = 0; n < ITERATIONS; n++)
    {
      if (n == 0)
	{
	  len = getpagesize ();
	  size = len + 512;
	  size1 = size;
	  size2 = size;
	  align1 = 512;
	  align2 = 512;
	}
      else
	{
	  if ((random () & 255) == 0)
	    size = 65536;
	  else
	    size = 768;
	  if (size > page_size)
	    size = page_size;
	  size1 = size;
	  size2 = size;
	  i = random ();
	  if (i & 3)
	    size -= 256;
	  if (i & 1)
	    size1 -= 256;
	  if (i & 2)
	    size2 -= 256;
	  if (i & 4)
	    {
	      len = random () % size;
	      align1 = size1 - len - (random () & 31);
	      align2 = size2 - len - (random () & 31);
	      if (align1 > size1)
		align1 = 0;
	      if (align2 > size2)
		align2 = 0;
	    }
	  else
	    {
	      align1 = random () & 63;
	      align2 = random () & 63;
	      len = random () % size;
	      if (align1 + len > size1)
		align1 = size1 - len;
	      if (align2 + len > size2)
		align2 = size2 - len;
	    }
	}
      p1 = buf1 + page_size - size1;
      p2 = buf2 + page_size - size2;
      c = random () & 255;
      j = align1 + len + 256;
      if (j > size1)
	j = size1;
      for (i = 0; i < j; ++i)
	p1[i] = random () & 255;

      FOR_EACH_IMPL (impl, 1)
	{
	  j = align2 + len + 256;
	  if (j > size2)
	    j = size2;
	  memset (p2, c, j);
	  res = (unsigned char *) CALL (impl,
					(char *) (p2 + align2),
					(char *) (p1 + align1), len);
	  if (res != MEMCPY_RESULT (p2 + align2, len))
	    {
	      error (0, 0, "Iteration %zd - wrong result in function %s (%zd, %zd, %zd) %p != %p",
		     n, impl->name, align1, align2, len, res,
		     MEMCPY_RESULT (p2 + align2, len));
	      ret = 1;
	    }
	  for (i = 0; i < align2; ++i)
	    {
	      if (p2[i] != c)
		{
		  error (0, 0, "Iteration %zd - garbage before, %s (%zd, %zd, %zd)",
			 n, impl->name, align1, align2, len);
		  ret = 1;
		  break;
		}
	    }
	  for (i = align2 + len; i < j; ++i)
	    {
	      if (p2[i] != c)
		{
		  error (0, 0, "Iteration %zd - garbage after, %s (%zd, %zd, %zd)",
			 n, impl->name, align1, align2, len);
		  ret = 1;
		  break;
		}
	    }
	  if (memcmp (p1 + align1, p2 + align2, len))
	    {
	      error (0, 0, "Iteration %zd - different strings, %s (%zd, %zd, %zd)",
		     n, impl->name, align1, align2, len);
	      ret = 1;
	    }
	}
    }
}

static void
do_test1 (size_t size)
{
  void *large_buf;
  large_buf = mmap (NULL, size * 2 + page_size, PROT_READ | PROT_WRITE,
		    MAP_PRIVATE | MAP_ANON, -1, 0);
  if (large_buf == MAP_FAILED)
    {
      puts ("Failed to allocat large_buf, skipping do_test1");
      return;
    }

  if (mprotect (large_buf + size, page_size, PROT_NONE))
    error (EXIT_FAILURE, errno, "mprotect failed");

  size_t arrary_size = size / sizeof (uint32_t);
  uint32_t *dest = large_buf;
  uint32_t *src = large_buf + size + page_size;
  size_t i;
  size_t repeats;
  for(repeats = 0; repeats < 2; repeats++)
    {
      for (i = 0; i < arrary_size; i++)
        src[i] = (uint32_t) i;

      FOR_EACH_IMPL (impl, 0)
        {
            printf ("\t\tRunning: %s\n", impl->name);
          memset (dest, -1, size);
          CALL (impl, (char *) dest, (char *) src, size);
          for (i = 0; i < arrary_size; i++)
        if (dest[i] != src[i])
          {
            error (0, 0,
               "Wrong result in function %s dst \"%p\" src \"%p\" offset \"%zd\"",
               impl->name, dest, src, i);
            ret = 1;
            munmap ((void *) large_buf, size * 2 + page_size);
            return;
          }
        }
      dest = src;
      src = large_buf;
    }
  munmap ((void *) large_buf, size * 2 + page_size);
}

int
test_main (void)
{
  size_t i;

  test_init ();

  printf ("%23s", "");
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (i = 0; i < 18; ++i)
    {
      do_test (0, 0, 1 << i);
      do_test (i, 0, 1 << i);
      do_test (0, i, 1 << i);
      do_test (i, i, 1 << i);
    }
  for (i = 0; i < 32; ++i)
    {
      do_test (0, 0, i);
      do_test (i, 0, i);
      do_test (0, i, i);
      do_test (i, i, i);
    }

  for (i = 3; i < 32; ++i)
    {
      if ((i & (i - 1)) == 0)
	continue;
      do_test (0, 0, 16 * i);
      do_test (i, 0, 16 * i);
      do_test (0, i, 16 * i);
      do_test (i, i, 16 * i);
    }

  for (i = 19; i <= 25; ++i)
    {
      do_test (255, 0, 1 << i);
      do_test (0, 255, i);
      do_test (0, 4000, i);
    }

  do_test (0, 0, getpagesize ());

  do_random_tests ();

  do_test1 (0x100000);
  do_test1 (0x2000000);
  return ret;
}

#include <support/test-driver.c>
