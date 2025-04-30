/* strcspn with SSE4.2 intrinsics
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

/* We use 0x2:
	_SIDD_SBYTE_OPS
	| _SIDD_CMP_EQUAL_ANY
	| _SIDD_POSITIVE_POLARITY
	| _SIDD_LEAST_SIGNIFICANT
   on pcmpistri to compare xmm/mem128

   0 1 2 3 4 5 6 7 8 9 A B C D E F
   X X X X X X X X X X X X X X X X

   against xmm

   0 1 2 3 4 5 6 7 8 9 A B C D E F
   A A A A A A A A A A A A A A A A

   to find out if the first 16byte data element has any byte A and
   the offset of the first byte.  There are 3 cases:

   1. The first 16byte data element has the byte A at the offset X.
   2. The first 16byte data element has EOS and doesn't have the byte A.
   3. The first 16byte data element is valid and doesn't have the byte A.

   Here is the table of ECX, CFlag, ZFlag and SFlag for 2 cases:

    1		 X	  1	 0/1	  0
    2		16	  0	  1	  0
    3		16	  0	  0	  0

   We exit from the loop for cases 1 and 2 with jbe which branches
   when either CFlag or ZFlag is 1.  If CFlag == 1, ECX has the offset
   X for case 1.  */

#ifndef STRCSPN_SSE2
# define STRCSPN_SSE2 __strcspn_sse2
# define STRCSPN_SSE42 __strcspn_sse42
#endif

#ifdef USE_AS_STRPBRK
# define RETURN(val1, val2) return val1
#else
# define RETURN(val1, val2) return val2
#endif

extern
#ifdef USE_AS_STRPBRK
char *
#else
size_t
#endif
STRCSPN_SSE2 (const char *, const char *) attribute_hidden;


#ifdef USE_AS_STRPBRK
char *
#else
size_t
#endif
__attribute__ ((section (".text.sse4.2")))
STRCSPN_SSE42 (const char *s, const char *a)
{
  if (*a == 0)
    RETURN (NULL, strlen (s));

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
	    return STRCSPN_SSE2 (s, a);

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
	    return STRCSPN_SSE2 (s, a);
	}
    }

  offset = (int) ((size_t) s & 15);
  if (offset != 0)
    {
      /* Check partial string.  */
      aligned = (const char *) ((size_t) s & -16L);
      __m128i value = _mm_load_si128 ((__m128i *) aligned);

      value = __m128i_shift_right (value, offset);

      int length = _mm_cmpistri (mask, value, 0x2);
      /* No need to check ZFlag since ZFlag is always 1.  */
      int cflag = _mm_cmpistrc (mask, value, 0x2);
      if (cflag)
	RETURN ((char *) (s + length), length);
      /* Find where the NULL terminator is.  */
      int index = _mm_cmpistri (value, value, 0x3a);
      if (index < 16 - offset)
	RETURN (NULL, index);
      aligned += 16;
    }
  else
    aligned = s;

  while (1)
    {
      __m128i value = _mm_load_si128 ((__m128i *) aligned);
      int index = _mm_cmpistri (mask, value, 0x2);
      int cflag = _mm_cmpistrc (mask, value, 0x2);
      int zflag = _mm_cmpistrz (mask, value, 0x2);
      if (cflag)
	RETURN ((char *) (aligned + index), (size_t) (aligned + index - s));
      if (zflag)
	RETURN (NULL,
		/* Find where the NULL terminator is.  */
		(size_t) (aligned + _mm_cmpistri (value, value, 0x3a) - s));
      aligned += 16;
    }
}
