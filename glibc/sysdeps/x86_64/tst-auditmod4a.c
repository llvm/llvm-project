/* Test case for x86-64 preserved registers in dynamic linker.  */

#ifdef __AVX__
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

__m256i
audit_test (__m256i x0, __m256i x1, __m256i x2, __m256i x3,
	    __m256i x4, __m256i x5, __m256i x6, __m256i x7)
{
  __m256i ymm;

  ymm = _mm256_set1_epi32 (1);
  if (memcmp (&ymm, &x0, sizeof (ymm)))
    abort ();

  ymm = _mm256_set1_epi32 (2);
  if (memcmp (&ymm, &x1, sizeof (ymm)))
    abort ();

  ymm = _mm256_set1_epi32 (3);
  if (memcmp (&ymm, &x2, sizeof (ymm)))
    abort ();

  ymm = _mm256_set1_epi32 (4);
  if (memcmp (&ymm, &x3, sizeof (ymm)))
    abort ();

  ymm = _mm256_set1_epi32 (5);
  if (memcmp (&ymm, &x4, sizeof (ymm)))
    abort ();

  ymm = _mm256_set1_epi32 (6);
  if (memcmp (&ymm, &x5, sizeof (ymm)))
    abort ();

  ymm = _mm256_set1_epi32 (7);
  if (memcmp (&ymm, &x6, sizeof (ymm)))
    abort ();

  ymm = _mm256_set1_epi32 (8);
  if (memcmp (&ymm, &x7, sizeof (ymm)))
    abort ();

  return _mm256_setzero_si256 ();
}
#endif
