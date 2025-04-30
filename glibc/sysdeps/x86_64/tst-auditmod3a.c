/* Test case for x86-64 preserved registers in dynamic linker.  */

#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>

__m128i
audit_test (__m128i x0, __m128i x1, __m128i x2, __m128i x3,
	    __m128i x4, __m128i x5, __m128i x6, __m128i x7)
{
  __m128i xmm = _mm_setzero_si128 ();

  if (memcmp (&xmm, &x0, sizeof (xmm))
      || memcmp (&xmm, &x1, sizeof (xmm))
      || memcmp (&xmm, &x2, sizeof (xmm))
      || memcmp (&xmm, &x3, sizeof (xmm))
      || memcmp (&xmm, &x4, sizeof (xmm))
      || memcmp (&xmm, &x5, sizeof (xmm))
      || memcmp (&xmm, &x6, sizeof (xmm))
      || memcmp (&xmm, &x7, sizeof (xmm)))
    abort ();

  return xmm;
}
