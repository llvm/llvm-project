/* Test and measure memmove functions.
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

#define TEST_MAIN
#ifdef TEST_BCOPY
# define TEST_NAME "bcopy"
#else
# define TEST_NAME "memmove"
#endif
#include "test-string.h"
#include <support/test-driver.h>

char *simple_memmove (char *, const char *, size_t);

#ifdef TEST_BCOPY
typedef void (*proto_t) (const char *, char *, size_t);
void simple_bcopy (const char *, char *, size_t);

IMPL (simple_bcopy, 0)
IMPL (bcopy, 1)

void
simple_bcopy (const char *src, char *dst, size_t n)
{
  simple_memmove (dst, src, n);
}
#else
typedef char *(*proto_t) (char *, const char *, size_t);

IMPL (simple_memmove, 0)
IMPL (memmove, 1)
#endif

char *
inhibit_loop_to_libcall
simple_memmove (char *dst, const char *src, size_t n)
{
  char *ret = dst;
  if (src < dst)
    {
      dst += n;
      src += n;
      while (n--)
	*--dst = *--src;
    }
  else
    while (n--)
      *dst++ = *src++;
  return ret;
}

static void
do_one_test (impl_t *impl, char *dst, char *src, const char *orig_src,
	     size_t len)
{
  /* This also clears the destination buffer set by the previous run.  */
  memcpy (src, orig_src, len);
#ifdef TEST_BCOPY
  CALL (impl, src, dst, len);
#else
  char *res;

  res = CALL (impl, dst, src, len);
  if (res != dst)
    {
      error (0, 0, "Wrong result in function %s %p %p", impl->name,
	     res, dst);
      ret = 1;
      return;
    }
#endif

  if (memcmp (dst, orig_src, len) != 0)
    {
      error (0, 0, "Wrong result in function %s dst \"%s\" src \"%s\"",
	     impl->name, dst, src);
      ret = 1;
      return;
    }
}

static void
do_test (size_t align1, size_t align2, size_t len)
{
  size_t i, j;
  char *s1, *s2;

  align1 &= 63;
  if (align1 + len >= page_size)
    return;

  align2 &= 63;
  if (align2 + len >= page_size)
    return;

  s1 = (char *) (buf1 + align1);
  s2 = (char *) (buf2 + align2);

  for (i = 0, j = 1; i < len; i++, j += 23)
    s1[i] = j;

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, s2, (char *) (buf2 + align1), s1, len);
}

static void
do_random_tests (void)
{
  size_t i, n, align1, align2, len, size;
  size_t srcstart, srcend, dststart, dstend;
  int c;
  unsigned char *p1, *p2;
#ifndef TEST_BCOPY
  unsigned char *res;
#endif

  for (n = 0; n < ITERATIONS; n++)
    {
      if ((random () & 255) == 0)
	size = 65536;
      else
	size = 512;
      if (size > page_size)
	size = page_size;
      if ((random () & 3) == 0)
	{
	  len = random () & (size - 1);
	  align1 = size - len - (random () & 31);
	  align2 = size - len - (random () & 31);
	  if (align1 > size)
	    align1 = 0;
	  if (align2 > size)
	    align2 = 0;
	}
      else
	{
	  align1 = random () & (size / 2 - 1);
	  align2 = random () & (size / 2 - 1);
	  len = random () & (size - 1);
	  if (align1 + len > size)
	    align1 = size - len;
	  if (align2 + len > size)
	    align2 = size - len;
	}

      p1 = buf1 + page_size - size;
      p2 = buf2 + page_size - size;
      c = random () & 255;
      srcend = align1 + len + 256;
      if (srcend > size)
	srcend = size;
      if (align1 > 256)
	srcstart = align1 - 256;
      else
	srcstart = 0;
      for (i = srcstart; i < srcend; ++i)
	p1[i] = random () & 255;
      dstend = align2 + len + 256;
      if (dstend > size)
	dstend = size;
      if (align2 > 256)
	dststart = align2 - 256;
      else
	dststart = 0;

      FOR_EACH_IMPL (impl, 1)
	{
	  memset (p2 + dststart, c, dstend - dststart);
	  memcpy (p2 + srcstart, p1 + srcstart, srcend - srcstart);
#ifdef TEST_BCOPY
	  CALL (impl, (char *) (p2 + align1), (char *) (p2 + align2), len);
#else
	  res = (unsigned char *) CALL (impl,
					(char *) (p2 + align2),
					(char *) (p2 + align1), len);
	  if (res != p2 + align2)
	    {
	      error (0, 0, "Iteration %zd - wrong result in function %s (%zd, %zd, %zd) %p != %p",
		     n, impl->name, align1, align2, len, res, p2 + align2);
	      ret = 1;
	    }
#endif
	  if (memcmp (p1 + align1, p2 + align2, len))
	    {
	      error (0, 0, "Iteration %zd - different strings, %s (%zd, %zd, %zd)",
		     n, impl->name, align1, align2, len);
	      ret = 1;
	    }
	  for (i = dststart; i < dstend; ++i)
	    {
	      if (i >= align2 && i < align2 + len)
		{
		  i = align2 + len - 1;
		  continue;
		}
	      if (i >= srcstart && i < srcend)
		{
		  i = srcend - 1;
		  continue;
		}
	      if (p2[i] != c)
		{
		  error (0, 0, "Iteration %zd - garbage in memset area, %s (%zd, %zd, %zd)",
			 n, impl->name, align1, align2, len);
		  ret = 1;
		  break;
		}
	    }

	  if (srcstart < align2
	      && memcmp (p2 + srcstart, p1 + srcstart,
			 (srcend > align2 ? align2 : srcend) - srcstart))
	    {
	      error (0, 0, "Iteration %zd - garbage before dst, %s (%zd, %zd, %zd)",
		     n, impl->name, align1, align2, len);
	      ret = 1;
	      break;
	    }

	  i = srcstart > align2 + len ? srcstart : align2 + len;
	  if (srcend > align2 + len
	      && memcmp (p2 + i, p1 + i, srcend - i))
	    {
	      error (0, 0, "Iteration %zd - garbage after dst, %s (%zd, %zd, %zd)",
		     n, impl->name, align1, align2, len);
	      ret = 1;
	      break;
	    }
	}
    }
}

static void
do_test2 (size_t offset)
{
  size_t size = 0x20000000;
  uint32_t * large_buf;

  large_buf = mmap ((void*) 0x70000000, size, PROT_READ | PROT_WRITE,
		    MAP_PRIVATE | MAP_ANON, -1, 0);

  if (large_buf == MAP_FAILED)
    error (EXIT_UNSUPPORTED, errno, "Large mmap failed");

  if ((uintptr_t) large_buf > 0x80000000 - 128
      || 0x80000000 - (uintptr_t) large_buf > 0x20000000)
    {
      error (0, 0, "Large mmap allocated improperly");
      ret = EXIT_UNSUPPORTED;
      munmap ((void *) large_buf, size);
      return;
    }

  size_t bytes_move = 0x80000000 - (uintptr_t) large_buf;
  if (bytes_move + offset * sizeof (uint32_t) > size)
    {
      munmap ((void *) large_buf, size);
      return;
    }
  size_t arr_size = bytes_move / sizeof (uint32_t);
  size_t i;
  size_t repeats;
  uint32_t * src = large_buf;
  uint32_t * dst = &large_buf[offset];
  for (repeats = 0; repeats < 2; ++repeats)
    {
      FOR_EACH_IMPL (impl, 0)
        {
          for (i = 0; i < arr_size; i++)
            src[i] = (uint32_t) i;


#ifdef TEST_BCOPY
          CALL (impl, (char *) src, (char *) dst, bytes_move);
#else
          CALL (impl, (char *) dst, (char *) src, bytes_move);
#endif

          for (i = 0; i < arr_size; i++)
	    {
	      if (dst[i] != (uint32_t) i)
		{
		  error (0, 0,
			 "Wrong result in function %s dst \"%p\" src \"%p\" offset \"%zd\"",
			 impl->name, dst, large_buf, i);
		  ret = 1;
		  munmap ((void *) large_buf, size);
		  return;
		}
	    }
	}
      src = dst;
      dst = large_buf;
    }

  munmap ((void *) large_buf, size);
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

  for (i = 0; i < 14; ++i)
    {
      do_test (0, 32, 1 << i);
      do_test (32, 0, 1 << i);
      do_test (0, i, 1 << i);
      do_test (i, 0, 1 << i);
    }

  for (i = 0; i < 32; ++i)
    {
      do_test (0, 32, i);
      do_test (32, 0, i);
      do_test (0, i, i);
      do_test (i, 0, i);
    }

  for (i = 3; i < 32; ++i)
    {
      if ((i & (i - 1)) == 0)
	continue;
      do_test (0, 32, 16 * i);
      do_test (32, 0, 16 * i);
      do_test (0, i, 16 * i);
      do_test (i, 0, 16 * i);
    }

  do_random_tests ();

  do_test2 (33);
  do_test2 (0x200000);
  do_test2 (0x4000000 - 1);
  do_test2 (0x4000000);
  return ret;
}

#include <support/test-driver.c>
