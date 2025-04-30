/* Test case for preserved AVX registers in dynamic linker, -mavx part.
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

extern __m256i audit_test (__m256i, __m256i, __m256i, __m256i,
			   __m256i, __m256i, __m256i, __m256i);

int
tst_audit4_aux (void)
{
#ifdef __AVX__
  __m256i ymm = _mm256_setzero_si256 ();
  __m256i ret = audit_test (ymm, ymm, ymm, ymm, ymm, ymm, ymm, ymm);
  ymm =	 _mm256_set1_epi32 (0x12349876);
  if (memcmp (&ymm, &ret, sizeof (ret)))
    abort ();
  return 0;
#else  /* __AVX__ */
  return 77;
#endif  /* __AVX__ */
}
