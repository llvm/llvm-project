/*===---- tmmintrin.h - SSSE3 intrinsics -----------------------------------===
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
 
#ifndef __TMMINTRIN_H
#define __TMMINTRIN_H

#ifndef __SSSE3__
#error "SSSE3 instruction set not enabled"
#else

#include <pmmintrin.h>

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_abs_pi8(__m64 a)
{
    return (__m64)__builtin_ia32_pabsb((__v8qi)a);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_abs_epi8(__m128i a)
{
    return (__m128i)__builtin_ia32_pabsb128((__v16qi)a);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_abs_pi16(__m64 a)
{
    return (__m64)__builtin_ia32_pabsw((__v4hi)a);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_abs_epi16(__m128i a)
{
    return (__m128i)__builtin_ia32_pabsw128((__v8hi)a);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_abs_pi32(__m64 a)
{
    return (__m64)__builtin_ia32_pabsd((__v2si)a);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_abs_epi32(__m128i a)
{
    return (__m128i)__builtin_ia32_pabsd128((__v4si)a);
}

#define _mm_alignr_epi8(a, b, n) __extension__ ({ \
  __m128i __a = (a); \
  __m128i __b = (b); \
  (__m128i)__builtin_ia32_palignr128((__v16qi)__a, (__v16qi)__b, (n)); })

#define _mm_alignr_pi8(a, b, n) __extension__ ({ \
  __m64 __a = (a); \
  __m64 __b = (b); \
  (__m64)__builtin_ia32_palignr((__v8qi)__a, (__v8qi)__b, (n)); })

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_hadd_epi16(__m128i a, __m128i b)
{
    return (__m128i)__builtin_ia32_phaddw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_hadd_epi32(__m128i a, __m128i b)
{
    return (__m128i)__builtin_ia32_phaddd128((__v4si)a, (__v4si)b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_hadd_pi16(__m64 a, __m64 b)
{
    return (__m64)__builtin_ia32_phaddw((__v4hi)a, (__v4hi)b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_hadd_pi32(__m64 a, __m64 b)
{
    return (__m64)__builtin_ia32_phaddd((__v2si)a, (__v2si)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_hadds_epi16(__m128i a, __m128i b)
{
    return (__m128i)__builtin_ia32_phaddsw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_hadds_pi16(__m64 a, __m64 b)
{
    return (__m64)__builtin_ia32_phaddsw((__v4hi)a, (__v4hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_hsub_epi16(__m128i a, __m128i b)
{
    return (__m128i)__builtin_ia32_phsubw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_hsub_epi32(__m128i a, __m128i b)
{
    return (__m128i)__builtin_ia32_phsubd128((__v4si)a, (__v4si)b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_hsub_pi16(__m64 a, __m64 b)
{
    return (__m64)__builtin_ia32_phsubw((__v4hi)a, (__v4hi)b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_hsub_pi32(__m64 a, __m64 b)
{
    return (__m64)__builtin_ia32_phsubd((__v2si)a, (__v2si)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_hsubs_epi16(__m128i a, __m128i b)
{
    return (__m128i)__builtin_ia32_phsubsw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_hsubs_pi16(__m64 a, __m64 b)
{
    return (__m64)__builtin_ia32_phsubsw((__v4hi)a, (__v4hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maddubs_epi16(__m128i a, __m128i b)
{
    return (__m128i)__builtin_ia32_pmaddubsw128((__v16qi)a, (__v16qi)b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_maddubs_pi16(__m64 a, __m64 b)
{
    return (__m64)__builtin_ia32_pmaddubsw((__v8qi)a, (__v8qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mulhrs_epi16(__m128i a, __m128i b)
{
    return (__m128i)__builtin_ia32_pmulhrsw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_mulhrs_pi16(__m64 a, __m64 b)
{
    return (__m64)__builtin_ia32_pmulhrsw((__v4hi)a, (__v4hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_shuffle_epi8(__m128i a, __m128i b)
{
    return (__m128i)__builtin_ia32_pshufb128((__v16qi)a, (__v16qi)b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_shuffle_pi8(__m64 a, __m64 b)
{
    return (__m64)__builtin_ia32_pshufb((__v8qi)a, (__v8qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sign_epi8(__m128i a, __m128i b)
{
    return (__m128i)__builtin_ia32_psignb128((__v16qi)a, (__v16qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sign_epi16(__m128i a, __m128i b)
{
    return (__m128i)__builtin_ia32_psignw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sign_epi32(__m128i a, __m128i b)
{
    return (__m128i)__builtin_ia32_psignd128((__v4si)a, (__v4si)b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sign_pi8(__m64 a, __m64 b)
{
    return (__m64)__builtin_ia32_psignb((__v8qi)a, (__v8qi)b);  
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sign_pi16(__m64 a, __m64 b)
{
    return (__m64)__builtin_ia32_psignw((__v4hi)a, (__v4hi)b);  
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sign_pi32(__m64 a, __m64 b)
{
    return (__m64)__builtin_ia32_psignd((__v2si)a, (__v2si)b);
}

#endif /* __SSSE3__ */

#endif /* __TMMINTRIN_H */
