/* Test case for preserved SSE registers in dynamic linker.
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

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

extern __m128i sse_test (__m128i, __m128i, __m128i, __m128i,
			 __m128i, __m128i, __m128i, __m128i);

static int
do_test (void)
{
  __m128i xmm0 = _mm_set1_epi32 (0);
  __m128i xmm1 = _mm_set1_epi32 (1);
  __m128i xmm2 = _mm_set1_epi32 (2);
  __m128i xmm3 = _mm_set1_epi32 (3);
  __m128i xmm4 = _mm_set1_epi32 (4);
  __m128i xmm5 = _mm_set1_epi32 (5);
  __m128i xmm6 = _mm_set1_epi32 (6);
  __m128i xmm7 = _mm_set1_epi32 (7);
  __m128i ret = sse_test (xmm0, xmm1, xmm2, xmm3,
			  xmm4, xmm5, xmm6, xmm7);
  xmm0 =  _mm_set1_epi32 (0x12349876);
  if (memcmp (&xmm0, &ret, sizeof (ret)))
    abort ();
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../../test-skeleton.c"
