/* Helper for variable shifts of SSE registers.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
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

#include <stdint.h>
#include <tmmintrin.h>

extern const int8_t ___m128i_shift_right[31] attribute_hidden;

static __inline__ __m128i
__m128i_shift_right (__m128i value, unsigned long int offset)
{
  return _mm_shuffle_epi8 (value,
			   _mm_loadu_si128 ((__m128i *) (___m128i_shift_right
							 + offset)));
}
