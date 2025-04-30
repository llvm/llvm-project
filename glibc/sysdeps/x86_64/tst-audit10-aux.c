/* Test case for preserved AVX512 registers in dynamic linker, -mavx512f part.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
tst_audit10_aux (void)
{
#ifdef __AVX512F__
  extern __m512i audit_test (__m512i, __m512i, __m512i, __m512i,
                             __m512i, __m512i, __m512i, __m512i);

  __m512i zmm = _mm512_setzero_si512 ();
  __m512i ret = audit_test (zmm, zmm, zmm, zmm, zmm, zmm, zmm, zmm);

  zmm = _mm512_set1_epi64 (0x12349876);

  if (memcmp (&zmm, &ret, sizeof (ret)))
    abort ();
  return 0;
#else /* __AVX512F__ */
  return 77;
#endif /* __AVX512F__ */
}
