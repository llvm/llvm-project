/* Test memrchr with size_t in the lower 32 bits of 64-bit register.
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

#define TEST_NAME "memrchr"
#include "test-size_t.h"

IMPL (memchr, 1)

typedef void * (*proto_t) (const void *, int, size_t);

static void *
__attribute__ ((noinline, noclone))
do_memrchr (parameter_t a, parameter_t b)
{
  return CALL (&b, a.p, (uintptr_t) b.p, a.len);
}

static int
test_main (void)
{
  test_init ();

  parameter_t src = { { page_size }, buf2 };
  parameter_t c = { { 0 }, (void *) (uintptr_t) 0x12 };

  int ret = 0;
  FOR_EACH_IMPL (impl, 0)
    {
      c.fn = impl->fn;
      void * res = do_memrchr (src, c);
      if (res)
	{
	  error (0, 0, "Wrong result in function %s: %p != NULL",
		 impl->name, res);
	  ret = 1;
	}
    }

  return ret ? EXIT_FAILURE : EXIT_SUCCESS;
}

#include <support/test-driver.c>
