/* Test sinl for pseudo-zeros and unnormals for ldbl-96 (bug 25487).
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <math.h>
#include <math_ldbl.h>
#include <stdint.h>

static int
do_test (void)
{
  for (int i = 0; i < 64; i++)
    {
      uint64_t sig = i == 63 ? 0 : 1ULL << i;
      long double ld;
      SET_LDOUBLE_WORDS (ld, 0x4141,
			 sig >> 32, sig & 0xffffffffULL);
      /* The requirement is that no stack overflow occurs when the
	 pseudo-zero or unnormal goes through range reduction.  */
      volatile long double ldr;
      ldr = sinl (ld);
      (void) ldr;
    }
  return 0;
}

#include <support/test-driver.c>
