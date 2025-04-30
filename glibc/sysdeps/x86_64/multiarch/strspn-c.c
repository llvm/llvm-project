/* strspn with SSE4.2 intrinsics
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
   Contributed by Intel Corporation.
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

#include <nmmintrin.h>
#include <string.h>
#include "varshift.h"

/* We use 0x12:
	_SIDD_SBYTE_OPS
	| _SIDD_CMP_EQUAL_ANY
	| _SIDD_NEGATIVE_POLARITY
	| _SIDD_LEAST_SIGNIFICANT
   on pcmpistri to compare xmm/mem128

   0 1 2 3 4 5 6 7 8 9 A B C D E F
   X X X X X X X X X X X X X X X X

   against xmm

   0 1 2 3 4 5 6 7 8 9 A B C D E F
   A A A A A A A A A A A A A A A A

   to find out if the first 16byte data element has any non-A byte and
   the offset of the first byte.  There are 2 cases:

   1. The first 16byte data element has the non-A byte, including
      EOS, at the offset X.
   2. The first 16byte data element is valid and doesn't have the non-A
      byte.

   Here is the table of ECX, CFlag, ZFlag and SFlag for 2 cases:

   case		ECX	CFlag	ZFlag	SFlag
    1		 X	  1	 0/1	  0
    2		16	  0	  0	  0

   We exit from the loop for case 1.  */

extern size_t __strspn_sse2 (const char *, const char *) attribute_hidden;


size_t
__attribute__ ((section (".text.sse4.2")))
__strspn_sse42 (const char *s, const char *a)
{
  if (*a == 0)
    return 0;

  const char *aligned;
  __m128i mask;
  int offset = (int) ((size_t) a & 15);
  if (offset != 0)
    {
      /* Load masks.  */
      aligned = (const char *) ((size_t) a & -16L);
      __m128i mask0 = _mm_load_si128 ((__m128i *) aligned);

      mask = __m128i_shift_right (mask0, offset);

      /* Find where the NULL terminator is.  */
      int length = _mm_cmpistri (mask, mask, 0x3a);
      if (length == 16 - offset)
	{
	  /* There is no NULL terminator.  */
	  __m128i mask1 = _mm_load_si128 ((__m128i *) (aligned + 16));
	  int index = _mm_cmpistri (mask1, mask1, 0x3a);
	  length += index;

	  /* Don't use SSE4.2 if the length of A > 16.  */
	  if (length > 16)
	    return __strspn_sse2 (s, a);

	  if (index != 0)
	    {
	      /* Combine mask0 and mask1.  We could play games with
		 palignr, but frankly this data should be in L1 now
		 so do the merge via an unaligned load.  */
	      mask = _mm_loadu_si128 ((__m128i *) a);
	    }
	}
    }
  else
    {
      /* A is aligned.  */
      mask = _mm_load_si128 ((__m128i *) a);

      /* Find where the NULL terminator is.  */
      int length = _mm_cmpistri (mask, mask, 0x3a);
      if (length == 16)
	{
	  /* There is no NULL terminator.  Don't use SSE4.2 if the length
	     of A > 16.  */
	  if (a[16] != 0)
	    return __strspn_sse2 (s, a);
	}
    }

  offset = (int) ((size_t) s & 15);
  if (offset != 0)
    {
      /* Check partial string.  */
      aligned = (const char *) ((size_t) s & -16L);
      __m128i value = _mm_load_si128 ((__m128i *) aligned);

      value = __m128i_shift_right (value, offset);

      int length = _mm_cmpistri (mask, value, 0x12);
      /* No need to check CFlag since it is always 1.  */
      if (length < 16 - offset)
	return length;
      /* Find where the NULL terminator is.  */
      int index = _mm_cmpistri (value, value, 0x3a);
      if (index < 16 - offset)
	return length;
      aligned += 16;
    }
  else
    aligned = s;

  while (1)
    {
      __m128i value = _mm_load_si128 ((__m128i *) aligned);
      int index = _mm_cmpistri (mask, value, 0x12);
      int cflag = _mm_cmpistrc (mask, value, 0x12);
      if (cflag)
	return (size_t) (aligned + index - s);
      aligned += 16;
    }
}
