/* Test case for preserved AVX registers in dynamic linker, -mavx part.
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
tst_avx_aux (void)
{
#ifdef __AVX__
  extern __m256i avx_test (__m256i, __m256i, __m256i, __m256i,
			   __m256i, __m256i, __m256i, __m256i);

  __m256i ymm0 = _mm256_set1_epi32 (0);
  __m256i ymm1 = _mm256_set1_epi32 (1);
  __m256i ymm2 = _mm256_set1_epi32 (2);
  __m256i ymm3 = _mm256_set1_epi32 (3);
  __m256i ymm4 = _mm256_set1_epi32 (4);
  __m256i ymm5 = _mm256_set1_epi32 (5);
  __m256i ymm6 = _mm256_set1_epi32 (6);
  __m256i ymm7 = _mm256_set1_epi32 (7);
  __m256i ret = avx_test (ymm0, ymm1, ymm2, ymm3,
			  ymm4, ymm5, ymm6, ymm7);
  ymm0 =  _mm256_set1_epi32 (0x12349876);
  if (memcmp (&ymm0, &ret, sizeof (ret)))
    abort ();
  return 0;
#else  /* __AVX__ */
  return 77;
#endif  /* __AVX__ */
}
