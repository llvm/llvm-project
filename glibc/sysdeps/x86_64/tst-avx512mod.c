/* Test case for x86-64 preserved AVX512 registers in dynamic linker.  */

#ifdef __AVX512F__
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

__m512i
avx512_test (__m512i x0, __m512i x1, __m512i x2, __m512i x3,
	     __m512i x4, __m512i x5, __m512i x6, __m512i x7)
{
  __m512i zmm;

  zmm = _mm512_set1_epi32 (0);
  if (memcmp (&zmm, &x0, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi32 (1);
  if (memcmp (&zmm, &x1, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi32 (2);
  if (memcmp (&zmm, &x2, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi32 (3);
  if (memcmp (&zmm, &x3, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi32 (4);
  if (memcmp (&zmm, &x4, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi32 (5);
  if (memcmp (&zmm, &x5, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi32 (6);
  if (memcmp (&zmm, &x6, sizeof (zmm)))
    abort ();

  zmm = _mm512_set1_epi32 (7);
  if (memcmp (&zmm, &x7, sizeof (zmm)))
    abort ();

  return _mm512_set1_epi32 (0x12349876);
}
#endif
