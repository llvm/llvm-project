/* Test memset with size_t in the lower 32 bits of 64-bit register.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#ifdef WIDE
# define TEST_NAME "wmemset"
#else
# define TEST_NAME "memset"
#endif /* WIDE */

#include "test-size_t.h"

#ifdef WIDE
# include <wchar.h>
# define MEMSET wmemset
# define CHAR wchar_t
#else
# define MEMSET memset
# define CHAR char
#endif /* WIDE */

IMPL (MEMSET, 1)

typedef CHAR *(*proto_t) (CHAR *, int, size_t);

static void *
__attribute__ ((noinline, noclone))
do_memset (parameter_t a, parameter_t b)
{
  return CALL (&b, a.p, (uintptr_t) b.p, a.len);
}

static int
test_main (void)
{
  test_init ();

  CHAR ch = 0x23;
  parameter_t src = { { page_size / sizeof (CHAR) }, buf2 };
  parameter_t c = { { 0 }, (void *) (uintptr_t) ch };

  int ret = 0;
  FOR_EACH_IMPL (impl, 0)
    {
      c.fn = impl->fn;
      CHAR *p = (CHAR *) do_memset (src, c);
      size_t i;
      for (i = 0; i < src.len; i++)
	if (p[i] != ch)
	  {
	    error (0, 0, "Wrong result in function %s", impl->name);
	    ret = 1;
	  }
    }

  return ret ? EXIT_FAILURE : EXIT_SUCCESS;
}

#include <support/test-driver.c>
