/* Test strncpy functions.
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

#ifdef WIDE
# include <wchar.h>
# define CHAR wchar_t
# define UCHAR wchar_t
# define BIG_CHAR WCHAR_MAX
# define SMALL_CHAR 1273
# define MEMCMP wmemcmp
# define MEMSET wmemset
# define STRNLEN wcsnlen
#else
# define CHAR char
# define UCHAR unsigned char
# define BIG_CHAR CHAR_MAX
# define SMALL_CHAR 127
# define MEMCMP memcmp
# define MEMSET memset
# define STRNLEN strnlen
#endif /* !WIDE */


#ifndef STRNCPY_RESULT
# define STRNCPY_RESULT(dst, len, n) dst
# define TEST_MAIN
# ifndef WIDE
#  define TEST_NAME "strncpy"
# else
#  define TEST_NAME "wcsncpy"
# endif /* WIDE */
# include "test-string.h"
# ifndef WIDE
#  define SIMPLE_STRNCPY simple_strncpy
#  define STUPID_STRNCPY stupid_strncpy
#  define STRNCPY strncpy
# else
#  define SIMPLE_STRNCPY simple_wcsncpy
#  define STUPID_STRNCPY stupid_wcsncpy
#  define STRNCPY wcsncpy
# endif /* WIDE */

CHAR *SIMPLE_STRNCPY (CHAR *, const CHAR *, size_t);
CHAR *STUPID_STRNCPY (CHAR *, const CHAR *, size_t);

IMPL (STUPID_STRNCPY, 0)
IMPL (SIMPLE_STRNCPY, 0)
IMPL (STRNCPY, 1)

CHAR *
SIMPLE_STRNCPY (CHAR *dst, const CHAR *src, size_t n)
{
  CHAR *ret = dst;
  while (n--)
    if ((*dst++ = *src++) == '\0')
      {
	while (n--)
	  *dst++ = '\0';
	return ret;
      }
  return ret;
}

CHAR *
STUPID_STRNCPY (CHAR *dst, const CHAR *src, size_t n)
{
  size_t nc = STRNLEN (src, n);
  size_t i;

  for (i = 0; i < nc; ++i)
    dst[i] = src[i];
  for (; i < n; ++i)
    dst[i] = '\0';
  return dst;
}
#endif /* !STRNCPY_RESULT */

typedef CHAR *(*proto_t) (CHAR *, const CHAR *, size_t);

static void
do_one_test (impl_t *impl, CHAR *dst, const CHAR *src, size_t len, size_t n)
{
  if (CALL (impl, dst, src, n) != STRNCPY_RESULT (dst, len, n))
    {
      error (0, 0, "Wrong result in function %s %p %p", impl->name,
	     CALL (impl, dst, src, n), dst);
      ret = 1;
      return;
    }

  if (memcmp (dst, src, (len > n ? n : len) * sizeof (CHAR)) != 0)
    {
      error (0, 0, "Wrong result in function %s", impl->name);
      ret = 1;
      return;
    }

  if (n > len)
    {
      size_t i;

      for (i = len; i < n; ++i)
	if (dst [i] != '\0')
	  {
	    error (0, 0, "Wrong result in function %s", impl->name);
	    ret = 1;
	    return;
	  }
    }
}

static void
do_test (size_t align1, size_t align2, size_t len, size_t n, int max_char)
{
  size_t i;
  CHAR *s1, *s2;

/* For wcsncpy: align1 and align2 here mean alignment not in bytes,
   but in wchar_ts, in bytes it will equal to align * (sizeof (wchar_t)).  */
  align1 &= 7;
  if ((align1 + len) * sizeof (CHAR) >= page_size)
    return;

  align2 &= 7;
  if ((align2 + len) * sizeof (CHAR) >= page_size)
    return;

  s1 = (CHAR *) (buf1) + align1;
  s2 = (CHAR *) (buf2) + align2;

  for (i = 0; i < len; ++i)
    s1[i] = 32 + 23 * i % (max_char - 32);
  s1[len] = 0;
  for (i = len + 1; (i + align1) * sizeof (CHAR) < page_size && i < len + 64;
       ++i)
    s1[i] = 32 + 32 * i % (max_char - 32);

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, s2, s1, len, n);
}

static void
do_page_tests (void)
{
  CHAR *s1, *s2;
  const size_t maxoffset = 64;

  /* Put s1 at the maxoffset from the edge of buf1's last page.  */
  s1 = (CHAR *) buf1 + BUF1PAGES * page_size / sizeof(CHAR) - maxoffset;
  /* s2 needs room to put a string with size of maxoffset + 1 at s2 +
     (maxoffset - 1).  */
  s2 = (CHAR *) buf2 + page_size / sizeof(CHAR) - maxoffset * 2;

  MEMSET (s1, 'a', maxoffset - 1);
  s1[maxoffset - 1] = '\0';

  /* Both strings are bounded to a page with read/write access and the next
     page is protected with PROT_NONE (meaning that any access outside of the
     page regions will trigger an invalid memory access).

     The loop copies the string s1 for all possible offsets up to maxoffset
     for both inputs with a size larger than s1 (so memory access outside the
     expected memory regions might trigger invalid access).  */

  for (size_t off1 = 0; off1 < maxoffset; off1++)
    {
      for (size_t off2 = 0; off2 < maxoffset; off2++)
	{
	  FOR_EACH_IMPL (impl, 0)
	    do_one_test (impl, s2 + off2, s1 + off1, maxoffset - off1 - 1,
			 maxoffset + (maxoffset - off2));
	}
    }
}

static void
do_random_tests (void)
{
  size_t i, j, n, align1, align2, len, size, mode;
  UCHAR *p1 = (UCHAR *) (buf1 + page_size) - 512;
  UCHAR *p2 = (UCHAR *) (buf2 + page_size) - 512;
  UCHAR *res;

  for (n = 0; n < ITERATIONS; n++)
    {
      /* For wcsncpy: align1 and align2 here mean align not in bytes,
	 but in wchar_ts, in bytes it will equal to align * (sizeof
	 (wchar_t)).  */

      mode = random ();
      if (mode & 1)
	{
	  size = random () & 255;
	  align1 = 512 - size - (random () & 15);
	  if (mode & 2)
	    align2 = align1 - (random () & 24);
	  else
	    align2 = align1 - (random () & 31);
	  if (mode & 4)
	    {
	      j = align1;
	      align1 = align2;
	      align2 = j;
	    }
	  if (mode & 8)
	    len = size - (random () & 31);
	  else
	    len = 512;
	  if (len >= 512)
	    len = random () & 511;
	}
      else
	{
	  align1 = random () & 31;
	  if (mode & 2)
	    align2 = random () & 31;
	  else
	    align2 = align1 + (random () & 24);
	  len = random () & 511;
	  j = align1;
	  if (align2 > j)
	    j = align2;
	  if (mode & 4)
	    {
	      size = random () & 511;
	      if (size + j > 512)
		size = 512 - j - (random () & 31);
	    }
	  else
	    size = 512 - j;
	  if ((mode & 8) && len + j >= 512)
	    len = 512 - j - (random () & 7);
	}
      j = len + align1 + 64;
      if (j > 512)
	j = 512;
      for (i = 0; i < j; i++)
	{
	  if (i == len + align1)
	    p1[i] = 0;
	  else
	    {
	      p1[i] = random () & BIG_CHAR;
	      if (i >= align1 && i < len + align1 && !p1[i])
		p1[i] = (random () & SMALL_CHAR) + 3;
	    }
	}

      FOR_EACH_IMPL (impl, 1)
	{
	  MEMSET (p2 - 64, '\1', 512 + 64);
	  res = (UCHAR *) CALL (impl, (CHAR *) (p2 + align2),
				(CHAR *) (p1 + align1), size);
	  if (res != STRNCPY_RESULT (p2 + align2, len, size))
	    {
	      error (0, 0, "Iteration %zd - wrong result in function %s (%zd, %zd, %zd) %p != %p",
		     n, impl->name, align1, align2, len, res,
		     STRNCPY_RESULT (p2 + align2, len, size));
	      ret = 1;
	    }
	  for (j = 0; j < align2 + 64; ++j)
	    {
	      if (p2[j - 64] != '\1')
		{
		  error (0, 0, "Iteration %zd - garbage before, %s (%zd, %zd, %zd)",
			 n, impl->name, align1, align2, len);
		  ret = 1;
		  break;
		}
	    }
	  j = align2 + len + 1;
	  if (size + align2 > j)
	    j = size + align2;
	  for (; j < 512; ++j)
	    {
	      if (p2[j] != '\1')
		{
		  error (0, 0, "Iteration %zd - garbage after, %s (%zd, %zd, %zd)",
			 n, impl->name, align1, align2, len);
		  ret = 1;
		  break;
		}
	    }
	  for (j = align2 + len + 1; j < align2 + size; ++j)
	    if (p2[j])
	      {
		error (0, 0, "Iteration %zd - garbage after size, %s (%zd, %zd, %zd)",
		       n, impl->name, align1, align2, len);
		ret = 1;
		break;
	      }
	  j = len + 1;
	  if (size < j)
	    j = size;
	  if (MEMCMP (p1 + align1, p2 + align2, j))
	    {
	      error (0, 0, "Iteration %zd - different strings, %s (%zd, %zd, %zd)",
		     n, impl->name, align1, align2, len);
	      ret = 1;
	    }
	}
    }
}

int
test_main (void)
{
  size_t i;

  test_init ();

  printf ("%28s", "");
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (i = 1; i < 8; ++i)
    {
      do_test (i, i, 16, 16, SMALL_CHAR);
      do_test (i, i, 16, 16, BIG_CHAR);
      do_test (i, 2 * i, 16, 16, SMALL_CHAR);
      do_test (2 * i, i, 16, 16, BIG_CHAR);
      do_test (8 - i, 2 * i, 1 << i, 2 << i, SMALL_CHAR);
      do_test (2 * i, 8 - i, 2 << i, 1 << i, SMALL_CHAR);
      do_test (8 - i, 2 * i, 1 << i, 2 << i, BIG_CHAR);
      do_test (2 * i, 8 - i, 2 << i, 1 << i, BIG_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 0, 4 << i, 8 << i, SMALL_CHAR);
      do_test (0, 0, 16 << i, 8 << i, SMALL_CHAR);
      do_test (8 - i, 2 * i, 4 << i, 8 << i, SMALL_CHAR);
      do_test (8 - i, 2 * i, 16 << i, 8 << i, SMALL_CHAR);
    }

  do_random_tests ();
  do_page_tests ();
  return ret;
}

#include <support/test-driver.c>
