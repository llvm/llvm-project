/* Test case for preserved AVX512 registers in dynamic linker,
   -mavx512 part.
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

int
tst_avx512_aux (void)
{
#ifdef __AVX512F__
  extern __m512i avx512_test (__m512i, __m512i, __m512i, __m512i,
			      __m512i, __m512i, __m512i, __m512i);

  __m512i zmm0 = _mm512_set1_epi32 (0);
  __m512i zmm1 = _mm512_set1_epi32 (1);
  __m512i zmm2 = _mm512_set1_epi32 (2);
  __m512i zmm3 = _mm512_set1_epi32 (3);
  __m512i zmm4 = _mm512_set1_epi32 (4);
  __m512i zmm5 = _mm512_set1_epi32 (5);
  __m512i zmm6 = _mm512_set1_epi32 (6);
  __m512i zmm7 = _mm512_set1_epi32 (7);
  __m512i ret = avx512_test (zmm0, zmm1, zmm2, zmm3,
			     zmm4, zmm5, zmm6, zmm7);
  zmm0 =  _mm512_set1_epi32 (0x12349876);
  if (memcmp (&zmm0, &ret, sizeof (ret)))
    abort ();
  return 0;
#else  /* __AVX512F__ */
  return 77;
#endif  /* __AVX512F__ */
}
