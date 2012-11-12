/*===---- pmmintrin.h - SSE3 intrinsics ------------------------------------===
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */
 
#ifndef __PMMINTRIN_H
#define __PMMINTRIN_H

#ifndef __SSE3__
#error "SSE3 instruction set not enabled"
#else

#include <emmintrin.h>

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_lddqu_si128(__m128i const *p)
{
  return (__m128i)__builtin_ia32_lddqu((char const *)p);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_addsub_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_addsubps(a, b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_hadd_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_haddps(a, b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_hsub_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_hsubps(a, b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_movehdup_ps(__m128 a)
{
  return __builtin_shufflevector(a, a, 1, 1, 3, 3);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_moveldup_ps(__m128 a)
{
  return __builtin_shufflevector(a, a, 0, 0, 2, 2);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_addsub_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_addsubpd(a, b);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_hadd_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_haddpd(a, b);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_hsub_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_hsubpd(a, b);
}

#define        _mm_loaddup_pd(dp)        _mm_load1_pd(dp)

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_movedup_pd(__m128d a)
{
  return __builtin_shufflevector(a, a, 0, 0);
}

#define _MM_DENORMALS_ZERO_ON   (0x0040)
#define _MM_DENORMALS_ZERO_OFF  (0x0000)

#define _MM_DENORMALS_ZERO_MASK (0x0040)

#define _MM_GET_DENORMALS_ZERO_MODE() (_mm_getcsr() & _MM_DENORMALS_ZERO_MASK)
#define _MM_SET_DENORMALS_ZERO_MODE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_DENORMALS_ZERO_MASK) | (x)))

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_monitor(void const *p, unsigned extensions, unsigned hints)
{
  __builtin_ia32_monitor((void *)p, extensions, hints);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_mwait(unsigned extensions, unsigned hints)
{
  __builtin_ia32_mwait(extensions, hints);
}

#endif /* __SSE3__ */

#endif /* __PMMINTRIN_H */
