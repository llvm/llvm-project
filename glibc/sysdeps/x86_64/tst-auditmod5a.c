/* Test case for x86-64 preserved registers in dynamic linker.  */

#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>

__m128i
audit_test (__m128i x0, __m128i x1, __m128i x2, __m128i x3,
	    __m128i x4, __m128i x5, __m128i x6, __m128i x7)
{
  __m128i xmm;

  xmm =  _mm_set1_epi32 (1);
  if (memcmp (&xmm, &x0, sizeof (xmm)))
    abort ();

  xmm =  _mm_set1_epi32 (2);
  if (memcmp (&xmm, &x1, sizeof (xmm)))
    abort ();

  xmm =  _mm_set1_epi32 (3);
  if (memcmp (&xmm, &x2, sizeof (xmm)))
    abort ();

  xmm =  _mm_set1_epi32 (4);
  if (memcmp (&xmm, &x3, sizeof (xmm)))
    abort ();

  xmm =  _mm_set1_epi32 (5);
  if (memcmp (&xmm, &x4, sizeof (xmm)))
    abort ();

  xmm =  _mm_set1_epi32 (6);
  if (memcmp (&xmm, &x5, sizeof (xmm)))
    abort ();

  xmm =  _mm_set1_epi32 (7);
  if (memcmp (&xmm, &x6, sizeof (xmm)))
    abort ();

  xmm =  _mm_set1_epi32 (8);
  if (memcmp (&xmm, &x7, sizeof (xmm)))
    abort ();

  return _mm_setzero_si128 ();
}
