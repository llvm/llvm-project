/* Test case for x86-64 preserved registers in dynamic linker.  */

#include <stdlib.h>
#include <string.h>

#include <emmintrin.h>

extern __m128i audit_test (__m128i, __m128i, __m128i, __m128i,
			   __m128i, __m128i, __m128i, __m128i);
static int
do_test (void)
{
  __m128i xmm = _mm_setzero_si128 ();
  __m128i ret = audit_test (xmm, xmm, xmm, xmm, xmm, xmm, xmm, xmm);

  if (memcmp (&xmm, &ret, sizeof (ret)))
    abort ();

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../../test-skeleton.c"
