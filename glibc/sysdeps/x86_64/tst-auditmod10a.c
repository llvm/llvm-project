/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

/* Test case for x86-64 preserved registers in dynamic linker.  */

#ifdef __AVX512F__
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

__m512i
audit_test (__m512i x0, __m512i x1, __m512i x2, __m512i x3,
	    __m512i x4, __m512i x5, __m512i x6, __m512i x7)
{
  __m512i zmm;

  zmm = _mm512_set1_epi64 (1);
  if (memcmp (&zmm, &x0, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi64 (2);
  if (memcmp (&zmm, &x1, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi64 (3);
  if (memcmp (&zmm, &x2, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi64 (4);
  if (memcmp (&zmm, &x3, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi64 (5);
  if (memcmp (&zmm, &x4, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi64 (6);
  if (memcmp (&zmm, &x5, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi64 (7);
  if (memcmp (&zmm, &x6, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi64 (8);
  if (memcmp (&zmm, &x7, sizeof (zmm)))
    abort ();

  return _mm512_setzero_si512 ();
}
#endif
