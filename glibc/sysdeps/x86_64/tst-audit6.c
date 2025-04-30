/* Test case for x86-64 preserved registers in dynamic linker.  */

#include <stdlib.h>
#include <string.h>
#include <cpuid.h>
#include <emmintrin.h>

extern __m128i audit_test (__m128i, __m128i, __m128i, __m128i,
			   __m128i, __m128i, __m128i, __m128i);


static int
avx_enabled (void)
{
  unsigned int eax, ebx, ecx, edx;

  if (__get_cpuid (1, &eax, &ebx, &ecx, &edx) == 0
      || (ecx & (bit_AVX | bit_OSXSAVE)) != (bit_AVX | bit_OSXSAVE))
    return 0;

  /* Check the OS has AVX and SSE saving enabled.  */
  asm ("xgetbv" : "=a" (eax), "=d" (edx) : "c" (0));

  return (eax & 6) == 6;
}


static int
do_test (void)
{
  /* Run AVX test only if AVX is supported.  */
  if (avx_enabled ())
    {
      __m128i xmm = _mm_setzero_si128 ();
      __m128i ret = audit_test (xmm, xmm, xmm, xmm, xmm, xmm, xmm, xmm);

      xmm = _mm_set1_epi32 (0x98abcdef);
      if (memcmp (&xmm, &ret, sizeof (ret)))
	abort ();
    }
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../../test-skeleton.c"
