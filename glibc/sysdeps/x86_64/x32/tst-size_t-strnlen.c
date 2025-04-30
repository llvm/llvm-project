/* Test strnlen with size_t in the lower 32 bits of 64-bit register.
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
# define TEST_NAME "wcsnlen"
#else
# define TEST_NAME "strnlen"
#endif /* WIDE */

#include "test-size_t.h"

#ifdef WIDE
# include <wchar.h>
# define STRNLEN wcsnlen
# define CHAR wchar_t
#else
# define STRNLEN strnlen
# define CHAR char
#endif /* WIDE */

IMPL (STRNLEN, 1)

typedef size_t (*proto_t) (const CHAR *, size_t);

static size_t
__attribute__ ((noinline, noclone))
do_strnlen (parameter_t a, parameter_t b)
{
  return CALL (&a, a.p, b.len);
}

static int
test_main (void)
{
  test_init ();

  size_t size = page_size / sizeof (CHAR);
  parameter_t src = { { 0 }, buf2 };
  parameter_t c = { { size }, (void *) (uintptr_t) 'a' };

  int ret = 0;
  FOR_EACH_IMPL (impl, 0)
    {
      src.fn = impl->fn;
      size_t res = do_strnlen (src, c);
      if (res != size)
	{
	  error (0, 0, "Wrong result in function %s: 0x%x != 0x%x",
		 impl->name, res, size);
	  ret = 1;
	}
    }

  return ret ? EXIT_FAILURE : EXIT_SUCCESS;
}

#include <support/test-driver.c>
