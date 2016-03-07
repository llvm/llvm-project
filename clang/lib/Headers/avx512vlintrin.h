/*===---- avx512vlintrin.h - AVX512VL intrinsics ---------------------------===
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

#ifndef __IMMINTRIN_H
#error "Never use <avx512vlintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512VLINTRIN_H
#define __AVX512VLINTRIN_H

#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__, __target__("avx512vl")))
#define __DEFAULT_FN_ATTRS_BOTH __attribute__((__always_inline__, __nodebug__, __target__("avx512vl, avx512bw")))

static  __inline __v2di __DEFAULT_FN_ATTRS
_mm_setzero_di(void) {
  return (__v2di){ 0, 0};
}

/* Integer compare */

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm_cmpeq_epi32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqd128_mask((__v4si)__a, (__v4si)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm_mask_cmpeq_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqd128_mask((__v4si)__a, (__v4si)__b,
                                                  __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmpeq_epu32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 0,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmpeq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 0,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm256_cmpeq_epi32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqd256_mask((__v8si)__a, (__v8si)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm256_mask_cmpeq_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqd256_mask((__v8si)__a, (__v8si)__b,
                                                  __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmpeq_epu32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 0,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmpeq_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 0,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm_cmpeq_epi64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqq128_mask((__v2di)__a, (__v2di)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm_mask_cmpeq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqq128_mask((__v2di)__a, (__v2di)__b,
                                                  __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmpeq_epu64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 0,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmpeq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 0,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm256_cmpeq_epi64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqq256_mask((__v4di)__a, (__v4di)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm256_mask_cmpeq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqq256_mask((__v4di)__a, (__v4di)__b,
                                                  __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmpeq_epu64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 0,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmpeq_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 0,
                                                __u);
}


static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmpge_epi32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 5,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmpge_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 5,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmpge_epu32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 5,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmpge_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 5,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmpge_epi32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 5,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmpge_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 5,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmpge_epu32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 5,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmpge_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 5,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmpge_epi64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 5,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmpge_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 5,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmpge_epu64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 5,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmpge_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 5,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmpge_epi64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 5,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmpge_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 5,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmpge_epu64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 5,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmpge_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 5,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm_cmpgt_epi32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtd128_mask((__v4si)__a, (__v4si)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm_mask_cmpgt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtd128_mask((__v4si)__a, (__v4si)__b,
                                                  __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmpgt_epu32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 6,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmpgt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 6,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm256_cmpgt_epi32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtd256_mask((__v8si)__a, (__v8si)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm256_mask_cmpgt_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtd256_mask((__v8si)__a, (__v8si)__b,
                                                  __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmpgt_epu32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 6,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmpgt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 6,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm_cmpgt_epi64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtq128_mask((__v2di)__a, (__v2di)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm_mask_cmpgt_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtq128_mask((__v2di)__a, (__v2di)__b,
                                                  __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmpgt_epu64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 6,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmpgt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 6,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm256_cmpgt_epi64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtq256_mask((__v4di)__a, (__v4di)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS_BOTH
_mm256_mask_cmpgt_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtq256_mask((__v4di)__a, (__v4di)__b,
                                                  __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmpgt_epu64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 6,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmpgt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 6,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmple_epi32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 2,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmple_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 2,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmple_epu32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 2,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmple_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 2,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmple_epi32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 2,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmple_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 2,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmple_epu32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 2,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmple_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 2,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmple_epi64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 2,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmple_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 2,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmple_epu64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 2,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmple_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 2,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmple_epi64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 2,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmple_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 2,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmple_epu64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 2,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmple_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 2,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmplt_epi32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 1,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmplt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 1,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmplt_epu32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 1,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmplt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 1,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmplt_epi32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 1,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmplt_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 1,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmplt_epu32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 1,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmplt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 1,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmplt_epi64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 1,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmplt_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 1,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmplt_epu64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 1,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmplt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 1,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmplt_epi64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 1,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmplt_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 1,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmplt_epu64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 1,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmplt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 1,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmpneq_epi32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 4,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmpneq_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 4,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmpneq_epu32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 4,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmpneq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 4,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmpneq_epi32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 4,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmpneq_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 4,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmpneq_epu32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 4,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmpneq_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 4,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmpneq_epi64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 4,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmpneq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 4,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_cmpneq_epu64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 4,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm_mask_cmpneq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 4,
                                                __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmpneq_epi64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 4,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmpneq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 4,
                                               __u);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_cmpneq_epu64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 4,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __DEFAULT_FN_ATTRS
_mm256_mask_cmpneq_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 4,
                                                __u);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_add_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_paddd256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_add_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_paddd256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_add_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_paddq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_add_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_paddq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_sub_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_psubd256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_sub_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_psubd256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_sub_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_psubq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_sub_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_psubq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_add_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_paddd128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_add_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_paddd128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_add_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_paddq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_add_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_paddq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_sub_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_psubd128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_sub_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_psubd128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_sub_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_psubq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_sub_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_psubq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_mul_epi32 (__m256i __W, __mmask8 __M, __m256i __X,
           __m256i __Y)
{
  return (__m256i) __builtin_ia32_pmuldq256_mask ((__v8si) __X,
              (__v8si) __Y,
              (__v4di) __W, __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_mul_epi32 (__mmask8 __M, __m256i __X, __m256i __Y)
{
  return (__m256i) __builtin_ia32_pmuldq256_mask ((__v8si) __X,
              (__v8si) __Y,
              (__v4di)
              _mm256_setzero_si256 (),
              __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_mul_epi32 (__m128i __W, __mmask8 __M, __m128i __X,
        __m128i __Y)
{
  return (__m128i) __builtin_ia32_pmuldq128_mask ((__v4si) __X,
              (__v4si) __Y,
              (__v2di) __W, __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_mul_epi32 (__mmask8 __M, __m128i __X, __m128i __Y)
{
  return (__m128i) __builtin_ia32_pmuldq128_mask ((__v4si) __X,
              (__v4si) __Y,
              (__v2di)
              _mm_setzero_si128 (),
              __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_mul_epu32 (__m256i __W, __mmask8 __M, __m256i __X,
           __m256i __Y)
{
  return (__m256i) __builtin_ia32_pmuludq256_mask ((__v8si) __X,
               (__v8si) __Y,
               (__v4di) __W, __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_mul_epu32 (__mmask8 __M, __m256i __X, __m256i __Y)
{
  return (__m256i) __builtin_ia32_pmuludq256_mask ((__v8si) __X,
               (__v8si) __Y,
               (__v4di)
               _mm256_setzero_si256 (),
               __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_mul_epu32 (__m128i __W, __mmask8 __M, __m128i __X,
        __m128i __Y)
{
  return (__m128i) __builtin_ia32_pmuludq128_mask ((__v4si) __X,
               (__v4si) __Y,
               (__v2di) __W, __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_mul_epu32 (__mmask8 __M, __m128i __X, __m128i __Y)
{
  return (__m128i) __builtin_ia32_pmuludq128_mask ((__v4si) __X,
               (__v4si) __Y,
               (__v2di)
               _mm_setzero_si128 (),
               __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_mullo_epi32 (__mmask8 __M, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pmulld256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si)
              _mm256_setzero_si256 (),
              __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_mullo_epi32 (__m256i __W, __mmask8 __M, __m256i __A,
       __m256i __B)
{
  return (__m256i) __builtin_ia32_pmulld256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si) __W, __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_mullo_epi32 (__mmask8 __M, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pmulld128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si)
              _mm_setzero_si128 (),
              __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_mullo_epi32 (__m128i __W, __mmask16 __M, __m128i __A,
          __m128i __B)
{
  return (__m128i) __builtin_ia32_pmulld128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si) __W, __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_and_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_pandd256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_and_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pandd256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_and_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pandd128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_and_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pandd128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_andnot_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
        __m256i __B)
{
  return (__m256i) __builtin_ia32_pandnd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si) __W,
              (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_andnot_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pandnd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si)
              _mm256_setzero_si256 (),
              (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_andnot_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
           __m128i __B)
{
  return (__m128i) __builtin_ia32_pandnd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si) __W,
              (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_andnot_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pandnd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si)
              _mm_setzero_si128 (),
              (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_or_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
          __m256i __B)
{
  return (__m256i) __builtin_ia32_pord256_mask ((__v8si) __A,
            (__v8si) __B,
            (__v8si) __W,
            (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_or_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pord256_mask ((__v8si) __A,
            (__v8si) __B,
            (__v8si)
            _mm256_setzero_si256 (),
            (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_or_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pord128_mask ((__v4si) __A,
            (__v4si) __B,
            (__v4si) __W,
            (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_or_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pord128_mask ((__v4si) __A,
            (__v4si) __B,
            (__v4si)
            _mm_setzero_si128 (),
            (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_xor_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_pxord256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_xor_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pxord256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_xor_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_pxord128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_xor_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pxord128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_and_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_pandq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di) __W, __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_and_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pandq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di)
             _mm256_setzero_pd (),
             __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_and_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_pandq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di) __W, __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_and_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pandq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di)
             _mm_setzero_pd (),
             __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_andnot_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
        __m256i __B)
{
  return (__m256i) __builtin_ia32_pandnq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di) __W, __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_andnot_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pandnq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_pd (),
              __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_andnot_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
           __m128i __B)
{
  return (__m128i) __builtin_ia32_pandnq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di) __W, __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_andnot_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pandnq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_pd (),
              __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_or_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
          __m256i __B)
{
  return (__m256i) __builtin_ia32_porq256_mask ((__v4di) __A,
            (__v4di) __B,
            (__v4di) __W,
            (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_or_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_porq256_mask ((__v4di) __A,
            (__v4di) __B,
            (__v4di)
            _mm256_setzero_si256 (),
            (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_or_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_porq128_mask ((__v2di) __A,
            (__v2di) __B,
            (__v2di) __W,
            (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_or_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_porq128_mask ((__v2di) __A,
            (__v2di) __B,
            (__v2di)
            _mm_setzero_si128 (),
            (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_xor_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_pxorq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_xor_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pxorq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_xor_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_pxorq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_xor_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pxorq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

#define _mm_cmp_epi32_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)(__m128i)(a), \
                                        (__v4si)(__m128i)(b), \
                                        (p), (__mmask8)-1); })

#define _mm_mask_cmp_epi32_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)(__m128i)(a), \
                                        (__v4si)(__m128i)(b), \
                                        (p), (__mmask8)(m)); })

#define _mm_cmp_epu32_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)(__m128i)(a), \
                                         (__v4si)(__m128i)(b), \
                                         (p), (__mmask8)-1); })

#define _mm_mask_cmp_epu32_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)(__m128i)(a), \
                                         (__v4si)(__m128i)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm256_cmp_epi32_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)(__m256i)(a), \
                                        (__v8si)(__m256i)(b), \
                                        (p), (__mmask8)-1); })

#define _mm256_mask_cmp_epi32_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)(__m256i)(a), \
                                        (__v8si)(__m256i)(b), \
                                        (p), (__mmask8)(m)); })

#define _mm256_cmp_epu32_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)(__m256i)(a), \
                                         (__v8si)(__m256i)(b), \
                                         (p), (__mmask8)-1); })

#define _mm256_mask_cmp_epu32_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)(__m256i)(a), \
                                         (__v8si)(__m256i)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm_cmp_epi64_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)(__m128i)(a), \
                                        (__v2di)(__m128i)(b), \
                                        (p), (__mmask8)-1); })

#define _mm_mask_cmp_epi64_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)(__m128i)(a), \
                                        (__v2di)(__m128i)(b), \
                                        (p), (__mmask8)(m)); })

#define _mm_cmp_epu64_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)(__m128i)(a), \
                                         (__v2di)(__m128i)(b), \
                                         (p), (__mmask8)-1); })

#define _mm_mask_cmp_epu64_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)(__m128i)(a), \
                                         (__v2di)(__m128i)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm256_cmp_epi64_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)(__m256i)(a), \
                                        (__v4di)(__m256i)(b), \
                                        (p), (__mmask8)-1); })

#define _mm256_mask_cmp_epi64_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)(__m256i)(a), \
                                        (__v4di)(__m256i)(b), \
                                        (p), (__mmask8)(m)); })

#define _mm256_cmp_epu64_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)(__m256i)(a), \
                                         (__v4di)(__m256i)(b), \
                                         (p), (__mmask8)-1); })

#define _mm256_mask_cmp_epu64_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)(__m256i)(a), \
                                         (__v4di)(__m256i)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm256_cmp_ps_mask(a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpps256_mask((__v8sf)(__m256)(a), \
                                         (__v8sf)(__m256)(b), \
                                         (p), (__mmask8)-1); })

#define _mm256_mask_cmp_ps_mask(m, a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpps256_mask((__v8sf)(__m256)(a), \
                                         (__v8sf)(__m256)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm256_cmp_pd_mask(a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmppd256_mask((__v4df)(__m256)(a), \
                                         (__v4df)(__m256)(b), \
                                         (p), (__mmask8)-1); })

#define _mm256_mask_cmp_pd_mask(m, a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmppd256_mask((__v4df)(__m256)(a), \
                                         (__v4df)(__m256)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm128_cmp_ps_mask(a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpps128_mask((__v4sf)(__m128)(a), \
                                         (__v4sf)(__m128)(b), \
                                         (p), (__mmask8)-1); })

#define _mm128_mask_cmp_ps_mask(m, a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpps128_mask((__v4sf)(__m128)(a), \
                                         (__v4sf)(__m128)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm128_cmp_pd_mask(a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmppd128_mask((__v2df)(__m128)(a), \
                                         (__v2df)(__m128)(b), \
                                         (p), (__mmask8)-1); })

#define _mm128_mask_cmp_pd_mask(m, a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmppd128_mask((__v2df)(__m128)(a), \
                                         (__v2df)(__m128)(b), \
                                         (p), (__mmask8)(m)); })

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_fmadd_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C)
{
  return (__m128d) __builtin_ia32_vfmaddpd128_mask ((__v2df) __A,
                                                    (__v2df) __B,
                                                    (__v2df) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask3_fmadd_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U)
{
  return (__m128d) __builtin_ia32_vfmaddpd128_mask3 ((__v2df) __A,
                                                     (__v2df) __B,
                                                     (__v2df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_fmadd_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d) __builtin_ia32_vfmaddpd128_maskz ((__v2df) __A,
                                                     (__v2df) __B,
                                                     (__v2df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_fmsub_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C)
{
  return (__m128d) __builtin_ia32_vfmaddpd128_mask ((__v2df) __A,
                                                    (__v2df) __B,
                                                    -(__v2df) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_fmsub_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d) __builtin_ia32_vfmaddpd128_maskz ((__v2df) __A,
                                                     (__v2df) __B,
                                                     -(__v2df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask3_fnmadd_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U)
{
  return (__m128d) __builtin_ia32_vfmaddpd128_mask3 (-(__v2df) __A,
                                                     (__v2df) __B,
                                                     (__v2df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_fnmadd_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d) __builtin_ia32_vfmaddpd128_maskz (-(__v2df) __A,
                                                     (__v2df) __B,
                                                     (__v2df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_fnmsub_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d) __builtin_ia32_vfmaddpd128_maskz (-(__v2df) __A,
                                                     (__v2df) __B,
                                                     -(__v2df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_fmadd_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C)
{
  return (__m256d) __builtin_ia32_vfmaddpd256_mask ((__v4df) __A,
                                                    (__v4df) __B,
                                                    (__v4df) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask3_fmadd_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U)
{
  return (__m256d) __builtin_ia32_vfmaddpd256_mask3 ((__v4df) __A,
                                                     (__v4df) __B,
                                                     (__v4df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_fmadd_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d) __builtin_ia32_vfmaddpd256_maskz ((__v4df) __A,
                                                     (__v4df) __B,
                                                     (__v4df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_fmsub_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C)
{
  return (__m256d) __builtin_ia32_vfmaddpd256_mask ((__v4df) __A,
                                                    (__v4df) __B,
                                                    -(__v4df) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_fmsub_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d) __builtin_ia32_vfmaddpd256_maskz ((__v4df) __A,
                                                     (__v4df) __B,
                                                     -(__v4df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask3_fnmadd_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U)
{
  return (__m256d) __builtin_ia32_vfmaddpd256_mask3 (-(__v4df) __A,
                                                     (__v4df) __B,
                                                     (__v4df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_fnmadd_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d) __builtin_ia32_vfmaddpd256_maskz (-(__v4df) __A,
                                                     (__v4df) __B,
                                                     (__v4df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_fnmsub_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d) __builtin_ia32_vfmaddpd256_maskz (-(__v4df) __A,
                                                     (__v4df) __B,
                                                     -(__v4df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_fmadd_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C)
{
  return (__m128) __builtin_ia32_vfmaddps128_mask ((__v4sf) __A,
                                                   (__v4sf) __B,
                                                   (__v4sf) __C,
                                                   (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask3_fmadd_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U)
{
  return (__m128) __builtin_ia32_vfmaddps128_mask3 ((__v4sf) __A,
                                                    (__v4sf) __B,
                                                    (__v4sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_fmadd_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C)
{
  return (__m128) __builtin_ia32_vfmaddps128_maskz ((__v4sf) __A,
                                                    (__v4sf) __B,
                                                    (__v4sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_fmsub_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C)
{
  return (__m128) __builtin_ia32_vfmaddps128_mask ((__v4sf) __A,
                                                   (__v4sf) __B,
                                                   -(__v4sf) __C,
                                                   (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_fmsub_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C)
{
  return (__m128) __builtin_ia32_vfmaddps128_maskz ((__v4sf) __A,
                                                    (__v4sf) __B,
                                                    -(__v4sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask3_fnmadd_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U)
{
  return (__m128) __builtin_ia32_vfmaddps128_mask3 (-(__v4sf) __A,
                                                    (__v4sf) __B,
                                                    (__v4sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_fnmadd_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C)
{
  return (__m128) __builtin_ia32_vfmaddps128_maskz (-(__v4sf) __A,
                                                    (__v4sf) __B,
                                                    (__v4sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_fnmsub_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C)
{
  return (__m128) __builtin_ia32_vfmaddps128_maskz (-(__v4sf) __A,
                                                    (__v4sf) __B,
                                                    -(__v4sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_fmadd_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C)
{
  return (__m256) __builtin_ia32_vfmaddps256_mask ((__v8sf) __A,
                                                   (__v8sf) __B,
                                                   (__v8sf) __C,
                                                   (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask3_fmadd_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U)
{
  return (__m256) __builtin_ia32_vfmaddps256_mask3 ((__v8sf) __A,
                                                    (__v8sf) __B,
                                                    (__v8sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_fmadd_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C)
{
  return (__m256) __builtin_ia32_vfmaddps256_maskz ((__v8sf) __A,
                                                    (__v8sf) __B,
                                                    (__v8sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_fmsub_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C)
{
  return (__m256) __builtin_ia32_vfmaddps256_mask ((__v8sf) __A,
                                                   (__v8sf) __B,
                                                   -(__v8sf) __C,
                                                   (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_fmsub_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C)
{
  return (__m256) __builtin_ia32_vfmaddps256_maskz ((__v8sf) __A,
                                                    (__v8sf) __B,
                                                    -(__v8sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask3_fnmadd_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U)
{
  return (__m256) __builtin_ia32_vfmaddps256_mask3 (-(__v8sf) __A,
                                                    (__v8sf) __B,
                                                    (__v8sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_fnmadd_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C)
{
  return (__m256) __builtin_ia32_vfmaddps256_maskz (-(__v8sf) __A,
                                                    (__v8sf) __B,
                                                    (__v8sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_fnmsub_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C)
{
  return (__m256) __builtin_ia32_vfmaddps256_maskz (-(__v8sf) __A,
                                                    (__v8sf) __B,
                                                    -(__v8sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_fmaddsub_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C)
{
  return (__m128d) __builtin_ia32_vfmaddsubpd128_mask ((__v2df) __A,
                                                       (__v2df) __B,
                                                       (__v2df) __C,
                                                       (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask3_fmaddsub_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U)
{
  return (__m128d) __builtin_ia32_vfmaddsubpd128_mask3 ((__v2df) __A,
                                                        (__v2df) __B,
                                                        (__v2df) __C,
                                                        (__mmask8)
                                                        __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_fmaddsub_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d) __builtin_ia32_vfmaddsubpd128_maskz ((__v2df) __A,
                                                        (__v2df) __B,
                                                        (__v2df) __C,
                                                        (__mmask8)
                                                        __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_fmsubadd_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C)
{
  return (__m128d) __builtin_ia32_vfmaddsubpd128_mask ((__v2df) __A,
                                                       (__v2df) __B,
                                                       -(__v2df) __C,
                                                       (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_fmsubadd_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d) __builtin_ia32_vfmaddsubpd128_maskz ((__v2df) __A,
                                                        (__v2df) __B,
                                                        -(__v2df) __C,
                                                        (__mmask8)
                                                        __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_fmaddsub_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C)
{
  return (__m256d) __builtin_ia32_vfmaddsubpd256_mask ((__v4df) __A,
                                                       (__v4df) __B,
                                                       (__v4df) __C,
                                                       (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask3_fmaddsub_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U)
{
  return (__m256d) __builtin_ia32_vfmaddsubpd256_mask3 ((__v4df) __A,
                                                        (__v4df) __B,
                                                        (__v4df) __C,
                                                        (__mmask8)
                                                        __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_fmaddsub_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d) __builtin_ia32_vfmaddsubpd256_maskz ((__v4df) __A,
                                                        (__v4df) __B,
                                                        (__v4df) __C,
                                                        (__mmask8)
                                                        __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_fmsubadd_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C)
{
  return (__m256d) __builtin_ia32_vfmaddsubpd256_mask ((__v4df) __A,
                                                       (__v4df) __B,
                                                       -(__v4df) __C,
                                                       (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_fmsubadd_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d) __builtin_ia32_vfmaddsubpd256_maskz ((__v4df) __A,
                                                        (__v4df) __B,
                                                        -(__v4df) __C,
                                                        (__mmask8)
                                                        __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_fmaddsub_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C)
{
  return (__m128) __builtin_ia32_vfmaddsubps128_mask ((__v4sf) __A,
                                                      (__v4sf) __B,
                                                      (__v4sf) __C,
                                                      (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask3_fmaddsub_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U)
{
  return (__m128) __builtin_ia32_vfmaddsubps128_mask3 ((__v4sf) __A,
                                                       (__v4sf) __B,
                                                       (__v4sf) __C,
                                                       (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_fmaddsub_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C)
{
  return (__m128) __builtin_ia32_vfmaddsubps128_maskz ((__v4sf) __A,
                                                       (__v4sf) __B,
                                                       (__v4sf) __C,
                                                       (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_fmsubadd_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C)
{
  return (__m128) __builtin_ia32_vfmaddsubps128_mask ((__v4sf) __A,
                                                      (__v4sf) __B,
                                                      -(__v4sf) __C,
                                                      (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_fmsubadd_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C)
{
  return (__m128) __builtin_ia32_vfmaddsubps128_maskz ((__v4sf) __A,
                                                       (__v4sf) __B,
                                                       -(__v4sf) __C,
                                                       (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_fmaddsub_ps(__m256 __A, __mmask8 __U, __m256 __B,
                         __m256 __C)
{
  return (__m256) __builtin_ia32_vfmaddsubps256_mask ((__v8sf) __A,
                                                      (__v8sf) __B,
                                                      (__v8sf) __C,
                                                      (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask3_fmaddsub_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U)
{
  return (__m256) __builtin_ia32_vfmaddsubps256_mask3 ((__v8sf) __A,
                                                       (__v8sf) __B,
                                                       (__v8sf) __C,
                                                       (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_fmaddsub_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C)
{
  return (__m256) __builtin_ia32_vfmaddsubps256_maskz ((__v8sf) __A,
                                                       (__v8sf) __B,
                                                       (__v8sf) __C,
                                                       (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_fmsubadd_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C)
{
  return (__m256) __builtin_ia32_vfmaddsubps256_mask ((__v8sf) __A,
                                                      (__v8sf) __B,
                                                      -(__v8sf) __C,
                                                      (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_fmsubadd_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C)
{
  return (__m256) __builtin_ia32_vfmaddsubps256_maskz ((__v8sf) __A,
                                                       (__v8sf) __B,
                                                       -(__v8sf) __C,
                                                       (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask3_fmsub_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U)
{
  return (__m128d) __builtin_ia32_vfmsubpd128_mask3 ((__v2df) __A,
                                                     (__v2df) __B,
                                                     (__v2df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask3_fmsub_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U)
{
  return (__m256d) __builtin_ia32_vfmsubpd256_mask3 ((__v4df) __A,
                                                     (__v4df) __B,
                                                     (__v4df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask3_fmsub_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U)
{
  return (__m128) __builtin_ia32_vfmsubps128_mask3 ((__v4sf) __A,
                                                    (__v4sf) __B,
                                                    (__v4sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask3_fmsub_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U)
{
  return (__m256) __builtin_ia32_vfmsubps256_mask3 ((__v8sf) __A,
                                                    (__v8sf) __B,
                                                    (__v8sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask3_fmsubadd_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U)
{
  return (__m128d) __builtin_ia32_vfmsubaddpd128_mask3 ((__v2df) __A,
                                                        (__v2df) __B,
                                                        (__v2df) __C,
                                                        (__mmask8)
                                                        __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask3_fmsubadd_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U)
{
  return (__m256d) __builtin_ia32_vfmsubaddpd256_mask3 ((__v4df) __A,
                                                        (__v4df) __B,
                                                        (__v4df) __C,
                                                        (__mmask8)
                                                        __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask3_fmsubadd_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U)
{
  return (__m128) __builtin_ia32_vfmsubaddps128_mask3 ((__v4sf) __A,
                                                       (__v4sf) __B,
                                                       (__v4sf) __C,
                                                       (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask3_fmsubadd_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U)
{
  return (__m256) __builtin_ia32_vfmsubaddps256_mask3 ((__v8sf) __A,
                                                       (__v8sf) __B,
                                                       (__v8sf) __C,
                                                       (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_fnmadd_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C)
{
  return (__m128d) __builtin_ia32_vfnmaddpd128_mask ((__v2df) __A,
                                                     (__v2df) __B,
                                                     (__v2df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_fnmadd_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C)
{
  return (__m256d) __builtin_ia32_vfnmaddpd256_mask ((__v4df) __A,
                                                     (__v4df) __B,
                                                     (__v4df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_fnmadd_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C)
{
  return (__m128) __builtin_ia32_vfnmaddps128_mask ((__v4sf) __A,
                                                    (__v4sf) __B,
                                                    (__v4sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_fnmadd_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C)
{
  return (__m256) __builtin_ia32_vfnmaddps256_mask ((__v8sf) __A,
                                                    (__v8sf) __B,
                                                    (__v8sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_fnmsub_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C)
{
  return (__m128d) __builtin_ia32_vfnmsubpd128_mask ((__v2df) __A,
                                                     (__v2df) __B,
                                                     (__v2df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask3_fnmsub_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U)
{
  return (__m128d) __builtin_ia32_vfnmsubpd128_mask3 ((__v2df) __A,
                                                      (__v2df) __B,
                                                      (__v2df) __C,
                                                      (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_fnmsub_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C)
{
  return (__m256d) __builtin_ia32_vfnmsubpd256_mask ((__v4df) __A,
                                                     (__v4df) __B,
                                                     (__v4df) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask3_fnmsub_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U)
{
  return (__m256d) __builtin_ia32_vfnmsubpd256_mask3 ((__v4df) __A,
                                                      (__v4df) __B,
                                                      (__v4df) __C,
                                                      (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_fnmsub_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C)
{
  return (__m128) __builtin_ia32_vfnmsubps128_mask ((__v4sf) __A,
                                                    (__v4sf) __B,
                                                    (__v4sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask3_fnmsub_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U)
{
  return (__m128) __builtin_ia32_vfnmsubps128_mask3 ((__v4sf) __A,
                                                     (__v4sf) __B,
                                                     (__v4sf) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_fnmsub_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C)
{
  return (__m256) __builtin_ia32_vfnmsubps256_mask ((__v8sf) __A,
                                                    (__v8sf) __B,
                                                    (__v8sf) __C,
                                                    (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask3_fnmsub_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U)
{
  return (__m256) __builtin_ia32_vfnmsubps256_mask3 ((__v8sf) __A,
                                                     (__v8sf) __B,
                                                     (__v8sf) __C,
                                                     (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_add_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_addpd128_mask ((__v2df) __A,
             (__v2df) __B,
             (__v2df) __W,
             (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_add_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_addpd128_mask ((__v2df) __A,
             (__v2df) __B,
             (__v2df)
             _mm_setzero_pd (),
             (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_add_pd (__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  return (__m256d) __builtin_ia32_addpd256_mask ((__v4df) __A,
             (__v4df) __B,
             (__v4df) __W,
             (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_add_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  return (__m256d) __builtin_ia32_addpd256_mask ((__v4df) __A,
             (__v4df) __B,
             (__v4df)
             _mm256_setzero_pd (),
             (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_add_ps (__m128 __W, __mmask16 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_addps128_mask ((__v4sf) __A,
            (__v4sf) __B,
            (__v4sf) __W,
            (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_add_ps (__mmask16 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_addps128_mask ((__v4sf) __A,
            (__v4sf) __B,
            (__v4sf)
            _mm_setzero_ps (),
            (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_add_ps (__m256 __W, __mmask16 __U, __m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_addps256_mask ((__v8sf) __A,
            (__v8sf) __B,
            (__v8sf) __W,
            (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_add_ps (__mmask16 __U, __m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_addps256_mask ((__v8sf) __A,
            (__v8sf) __B,
            (__v8sf)
            _mm256_setzero_ps (),
            (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_blend_epi32 (__mmask8 __U, __m128i __A, __m128i __W) {
  return (__m128i) __builtin_ia32_blendmd_128_mask ((__v4si) __A,
                (__v4si) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_blend_epi32 (__mmask8 __U, __m256i __A, __m256i __W) {
  return (__m256i) __builtin_ia32_blendmd_256_mask ((__v8si) __A,
                (__v8si) __W,
                (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_blend_pd (__mmask8 __U, __m128d __A, __m128d __W) {
  return (__m128d) __builtin_ia32_blendmpd_128_mask ((__v2df) __A,
                 (__v2df) __W,
                 (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_blend_pd (__mmask8 __U, __m256d __A, __m256d __W) {
  return (__m256d) __builtin_ia32_blendmpd_256_mask ((__v4df) __A,
                 (__v4df) __W,
                 (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_blend_ps (__mmask8 __U, __m128 __A, __m128 __W) {
  return (__m128) __builtin_ia32_blendmps_128_mask ((__v4sf) __A,
                (__v4sf) __W,
                (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_blend_ps (__mmask8 __U, __m256 __A, __m256 __W) {
  return (__m256) __builtin_ia32_blendmps_256_mask ((__v8sf) __A,
                (__v8sf) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_blend_epi64 (__mmask8 __U, __m128i __A, __m128i __W) {
  return (__m128i) __builtin_ia32_blendmq_128_mask ((__v2di) __A,
                (__v2di) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_blend_epi64 (__mmask8 __U, __m256i __A, __m256i __W) {
  return (__m256i) __builtin_ia32_blendmq_256_mask ((__v4di) __A,
                (__v4di) __W,
                (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_compress_pd (__m128d __W, __mmask8 __U, __m128d __A) {
  return (__m128d) __builtin_ia32_compressdf128_mask ((__v2df) __A,
                  (__v2df) __W,
                  (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_compress_pd (__mmask8 __U, __m128d __A) {
  return (__m128d) __builtin_ia32_compressdf128_mask ((__v2df) __A,
                  (__v2df)
                  _mm_setzero_pd (),
                  (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_compress_pd (__m256d __W, __mmask8 __U, __m256d __A) {
  return (__m256d) __builtin_ia32_compressdf256_mask ((__v4df) __A,
                  (__v4df) __W,
                  (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_compress_pd (__mmask8 __U, __m256d __A) {
  return (__m256d) __builtin_ia32_compressdf256_mask ((__v4df) __A,
                  (__v4df)
                  _mm256_setzero_pd (),
                  (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_compress_epi64 (__m128i __W, __mmask8 __U, __m128i __A) {
  return (__m128i) __builtin_ia32_compressdi128_mask ((__v2di) __A,
                  (__v2di) __W,
                  (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_compress_epi64 (__mmask8 __U, __m128i __A) {
  return (__m128i) __builtin_ia32_compressdi128_mask ((__v2di) __A,
                  (__v2di)
                  _mm_setzero_si128 (),
                  (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_compress_epi64 (__m256i __W, __mmask8 __U, __m256i __A) {
  return (__m256i) __builtin_ia32_compressdi256_mask ((__v4di) __A,
                  (__v4di) __W,
                  (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_compress_epi64 (__mmask8 __U, __m256i __A) {
  return (__m256i) __builtin_ia32_compressdi256_mask ((__v4di) __A,
                  (__v4di)
                  _mm256_setzero_si256 (),
                  (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_compress_ps (__m128 __W, __mmask8 __U, __m128 __A) {
  return (__m128) __builtin_ia32_compresssf128_mask ((__v4sf) __A,
                 (__v4sf) __W,
                 (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_compress_ps (__mmask8 __U, __m128 __A) {
  return (__m128) __builtin_ia32_compresssf128_mask ((__v4sf) __A,
                 (__v4sf)
                 _mm_setzero_ps (),
                 (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_compress_ps (__m256 __W, __mmask8 __U, __m256 __A) {
  return (__m256) __builtin_ia32_compresssf256_mask ((__v8sf) __A,
                 (__v8sf) __W,
                 (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_compress_ps (__mmask8 __U, __m256 __A) {
  return (__m256) __builtin_ia32_compresssf256_mask ((__v8sf) __A,
                 (__v8sf)
                 _mm256_setzero_ps (),
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_compress_epi32 (__m128i __W, __mmask8 __U, __m128i __A) {
  return (__m128i) __builtin_ia32_compresssi128_mask ((__v4si) __A,
                  (__v4si) __W,
                  (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_compress_epi32 (__mmask8 __U, __m128i __A) {
  return (__m128i) __builtin_ia32_compresssi128_mask ((__v4si) __A,
                  (__v4si)
                  _mm_setzero_si128 (),
                  (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_compress_epi32 (__m256i __W, __mmask8 __U, __m256i __A) {
  return (__m256i) __builtin_ia32_compresssi256_mask ((__v8si) __A,
                  (__v8si) __W,
                  (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_compress_epi32 (__mmask8 __U, __m256i __A) {
  return (__m256i) __builtin_ia32_compresssi256_mask ((__v8si) __A,
                  (__v8si)
                  _mm256_setzero_si256 (),
                  (__mmask8) __U);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_mask_compressstoreu_pd (void *__P, __mmask8 __U, __m128d __A) {
  __builtin_ia32_compressstoredf128_mask ((__v2df *) __P,
            (__v2df) __A,
            (__mmask8) __U);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm256_mask_compressstoreu_pd (void *__P, __mmask8 __U, __m256d __A) {
  __builtin_ia32_compressstoredf256_mask ((__v4df *) __P,
            (__v4df) __A,
            (__mmask8) __U);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_mask_compressstoreu_epi64 (void *__P, __mmask8 __U, __m128i __A) {
  __builtin_ia32_compressstoredi128_mask ((__v2di *) __P,
            (__v2di) __A,
            (__mmask8) __U);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm256_mask_compressstoreu_epi64 (void *__P, __mmask8 __U, __m256i __A) {
  __builtin_ia32_compressstoredi256_mask ((__v4di *) __P,
            (__v4di) __A,
            (__mmask8) __U);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_mask_compressstoreu_ps (void *__P, __mmask8 __U, __m128 __A) {
  __builtin_ia32_compressstoresf128_mask ((__v4sf *) __P,
            (__v4sf) __A,
            (__mmask8) __U);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm256_mask_compressstoreu_ps (void *__P, __mmask8 __U, __m256 __A) {
  __builtin_ia32_compressstoresf256_mask ((__v8sf *) __P,
            (__v8sf) __A,
            (__mmask8) __U);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_mask_compressstoreu_epi32 (void *__P, __mmask8 __U, __m128i __A) {
  __builtin_ia32_compressstoresi128_mask ((__v4si *) __P,
            (__v4si) __A,
            (__mmask8) __U);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm256_mask_compressstoreu_epi32 (void *__P, __mmask8 __U, __m256i __A) {
  __builtin_ia32_compressstoresi256_mask ((__v8si *) __P,
            (__v8si) __A,
            (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_cvtepi32_pd (__m128d __W, __mmask8 __U, __m128i __A) {
  return (__m128d) __builtin_ia32_cvtdq2pd128_mask ((__v4si) __A,
                (__v2df) __W,
                (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_cvtepi32_pd (__mmask8 __U, __m128i __A) {
  return (__m128d) __builtin_ia32_cvtdq2pd128_mask ((__v4si) __A,
                (__v2df)
                _mm_setzero_pd (),
                (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_cvtepi32_pd (__m256d __W, __mmask8 __U, __m128i __A) {
  return (__m256d) __builtin_ia32_cvtdq2pd256_mask ((__v4si) __A,
                (__v4df) __W,
                (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepi32_pd (__mmask8 __U, __m128i __A) {
  return (__m256d) __builtin_ia32_cvtdq2pd256_mask ((__v4si) __A,
                (__v4df)
                _mm256_setzero_pd (),
                (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_cvtepi32_ps (__m128 __W, __mmask8 __U, __m128i __A) {
  return (__m128) __builtin_ia32_cvtdq2ps128_mask ((__v4si) __A,
               (__v4sf) __W,
               (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_cvtepi32_ps (__mmask16 __U, __m128i __A) {
  return (__m128) __builtin_ia32_cvtdq2ps128_mask ((__v4si) __A,
               (__v4sf)
               _mm_setzero_ps (),
               (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_cvtepi32_ps (__m256 __W, __mmask8 __U, __m256i __A) {
  return (__m256) __builtin_ia32_cvtdq2ps256_mask ((__v8si) __A,
               (__v8sf) __W,
               (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepi32_ps (__mmask16 __U, __m256i __A) {
  return (__m256) __builtin_ia32_cvtdq2ps256_mask ((__v8si) __A,
               (__v8sf)
               _mm256_setzero_ps (),
               (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtpd_epi32 (__m128i __W, __mmask8 __U, __m128d __A) {
  return (__m128i) __builtin_ia32_cvtpd2dq128_mask ((__v2df) __A,
                (__v4si) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtpd_epi32 (__mmask8 __U, __m128d __A) {
  return (__m128i) __builtin_ia32_cvtpd2dq128_mask ((__v2df) __A,
                (__v4si)
                _mm_setzero_si128 (),
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm256_mask_cvtpd_epi32 (__m128i __W, __mmask8 __U, __m256d __A) {
  return (__m128i) __builtin_ia32_cvtpd2dq256_mask ((__v4df) __A,
                (__v4si) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtpd_epi32 (__mmask8 __U, __m256d __A) {
  return (__m128i) __builtin_ia32_cvtpd2dq256_mask ((__v4df) __A,
                (__v4si)
                _mm_setzero_si128 (),
                (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_cvtpd_ps (__m128 __W, __mmask8 __U, __m128d __A) {
  return (__m128) __builtin_ia32_cvtpd2ps_mask ((__v2df) __A,
            (__v4sf) __W,
            (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_cvtpd_ps (__mmask8 __U, __m128d __A) {
  return (__m128) __builtin_ia32_cvtpd2ps_mask ((__v2df) __A,
            (__v4sf)
            _mm_setzero_ps (),
            (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm256_mask_cvtpd_ps (__m128 __W, __mmask8 __U, __m256d __A) {
  return (__m128) __builtin_ia32_cvtpd2ps256_mask ((__v4df) __A,
               (__v4sf) __W,
               (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm256_maskz_cvtpd_ps (__mmask8 __U, __m256d __A) {
  return (__m128) __builtin_ia32_cvtpd2ps256_mask ((__v4df) __A,
               (__v4sf)
               _mm_setzero_ps (),
               (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvtpd_epu32 (__m128d __A) {
  return (__m128i) __builtin_ia32_cvtpd2udq128_mask ((__v2df) __A,
                 (__v4si)
                 _mm_setzero_si128 (),
                 (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtpd_epu32 (__m128i __W, __mmask8 __U, __m128d __A) {
  return (__m128i) __builtin_ia32_cvtpd2udq128_mask ((__v2df) __A,
                 (__v4si) __W,
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtpd_epu32 (__mmask8 __U, __m128d __A) {
  return (__m128i) __builtin_ia32_cvtpd2udq128_mask ((__v2df) __A,
                 (__v4si)
                 _mm_setzero_si128 (),
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm256_cvtpd_epu32 (__m256d __A) {
  return (__m128i) __builtin_ia32_cvtpd2udq256_mask ((__v4df) __A,
                 (__v4si)
                 _mm_setzero_si128 (),
                 (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm256_mask_cvtpd_epu32 (__m128i __W, __mmask8 __U, __m256d __A) {
  return (__m128i) __builtin_ia32_cvtpd2udq256_mask ((__v4df) __A,
                 (__v4si) __W,
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtpd_epu32 (__mmask8 __U, __m256d __A) {
  return (__m128i) __builtin_ia32_cvtpd2udq256_mask ((__v4df) __A,
                 (__v4si)
                 _mm_setzero_si128 (),
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtps_epi32 (__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i) __builtin_ia32_cvtps2dq128_mask ((__v4sf) __A,
                (__v4si) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtps_epi32 (__mmask8 __U, __m128 __A) {
  return (__m128i) __builtin_ia32_cvtps2dq128_mask ((__v4sf) __A,
                (__v4si)
                _mm_setzero_si128 (),
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvtps_epi32 (__m256i __W, __mmask8 __U, __m256 __A) {
  return (__m256i) __builtin_ia32_cvtps2dq256_mask ((__v8sf) __A,
                (__v8si) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtps_epi32 (__mmask8 __U, __m256 __A) {
  return (__m256i) __builtin_ia32_cvtps2dq256_mask ((__v8sf) __A,
                (__v8si)
                _mm256_setzero_si256 (),
                (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_cvtps_pd (__m128d __W, __mmask8 __U, __m128 __A) {
  return (__m128d) __builtin_ia32_cvtps2pd128_mask ((__v4sf) __A,
                (__v2df) __W,
                (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_cvtps_pd (__mmask8 __U, __m128 __A) {
  return (__m128d) __builtin_ia32_cvtps2pd128_mask ((__v4sf) __A,
                (__v2df)
                _mm_setzero_pd (),
                (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_cvtps_pd (__m256d __W, __mmask8 __U, __m128 __A) {
  return (__m256d) __builtin_ia32_cvtps2pd256_mask ((__v4sf) __A,
                (__v4df) __W,
                (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_cvtps_pd (__mmask8 __U, __m128 __A) {
  return (__m256d) __builtin_ia32_cvtps2pd256_mask ((__v4sf) __A,
                (__v4df)
                _mm256_setzero_pd (),
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvtps_epu32 (__m128 __A) {
  return (__m128i) __builtin_ia32_cvtps2udq128_mask ((__v4sf) __A,
                 (__v4si)
                 _mm_setzero_si128 (),
                 (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtps_epu32 (__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i) __builtin_ia32_cvtps2udq128_mask ((__v4sf) __A,
                 (__v4si) __W,
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtps_epu32 (__mmask8 __U, __m128 __A) {
  return (__m128i) __builtin_ia32_cvtps2udq128_mask ((__v4sf) __A,
                 (__v4si)
                 _mm_setzero_si128 (),
                 (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_cvtps_epu32 (__m256 __A) {
  return (__m256i) __builtin_ia32_cvtps2udq256_mask ((__v8sf) __A,
                 (__v8si)
                 _mm256_setzero_si256 (),
                 (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvtps_epu32 (__m256i __W, __mmask8 __U, __m256 __A) {
  return (__m256i) __builtin_ia32_cvtps2udq256_mask ((__v8sf) __A,
                 (__v8si) __W,
                 (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtps_epu32 (__mmask8 __U, __m256 __A) {
  return (__m256i) __builtin_ia32_cvtps2udq256_mask ((__v8sf) __A,
                 (__v8si)
                 _mm256_setzero_si256 (),
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvttpd_epi32 (__m128i __W, __mmask8 __U, __m128d __A) {
  return (__m128i) __builtin_ia32_cvttpd2dq128_mask ((__v2df) __A,
                 (__v4si) __W,
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvttpd_epi32 (__mmask8 __U, __m128d __A) {
  return (__m128i) __builtin_ia32_cvttpd2dq128_mask ((__v2df) __A,
                 (__v4si)
                 _mm_setzero_si128 (),
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm256_mask_cvttpd_epi32 (__m128i __W, __mmask8 __U, __m256d __A) {
  return (__m128i) __builtin_ia32_cvttpd2dq256_mask ((__v4df) __A,
                 (__v4si) __W,
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm256_maskz_cvttpd_epi32 (__mmask8 __U, __m256d __A) {
  return (__m128i) __builtin_ia32_cvttpd2dq256_mask ((__v4df) __A,
                 (__v4si)
                 _mm_setzero_si128 (),
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvttpd_epu32 (__m128d __A) {
  return (__m128i) __builtin_ia32_cvttpd2udq128_mask ((__v2df) __A,
                  (__v4si)
                  _mm_setzero_si128 (),
                  (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvttpd_epu32 (__m128i __W, __mmask8 __U, __m128d __A) {
  return (__m128i) __builtin_ia32_cvttpd2udq128_mask ((__v2df) __A,
                  (__v4si) __W,
                  (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvttpd_epu32 (__mmask8 __U, __m128d __A) {
  return (__m128i) __builtin_ia32_cvttpd2udq128_mask ((__v2df) __A,
                  (__v4si)
                  _mm_setzero_si128 (),
                  (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm256_cvttpd_epu32 (__m256d __A) {
  return (__m128i) __builtin_ia32_cvttpd2udq256_mask ((__v4df) __A,
                  (__v4si)
                  _mm_setzero_si128 (),
                  (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm256_mask_cvttpd_epu32 (__m128i __W, __mmask8 __U, __m256d __A) {
  return (__m128i) __builtin_ia32_cvttpd2udq256_mask ((__v4df) __A,
                  (__v4si) __W,
                  (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm256_maskz_cvttpd_epu32 (__mmask8 __U, __m256d __A) {
  return (__m128i) __builtin_ia32_cvttpd2udq256_mask ((__v4df) __A,
                  (__v4si)
                  _mm_setzero_si128 (),
                  (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvttps_epi32 (__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i) __builtin_ia32_cvttps2dq128_mask ((__v4sf) __A,
                 (__v4si) __W,
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvttps_epi32 (__mmask8 __U, __m128 __A) {
  return (__m128i) __builtin_ia32_cvttps2dq128_mask ((__v4sf) __A,
                 (__v4si)
                 _mm_setzero_si128 (),
                 (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvttps_epi32 (__m256i __W, __mmask8 __U, __m256 __A) {
  return (__m256i) __builtin_ia32_cvttps2dq256_mask ((__v8sf) __A,
                 (__v8si) __W,
                 (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvttps_epi32 (__mmask8 __U, __m256 __A) {
  return (__m256i) __builtin_ia32_cvttps2dq256_mask ((__v8sf) __A,
                 (__v8si)
                 _mm256_setzero_si256 (),
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvttps_epu32 (__m128 __A) {
  return (__m128i) __builtin_ia32_cvttps2udq128_mask ((__v4sf) __A,
                  (__v4si)
                  _mm_setzero_si128 (),
                  (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvttps_epu32 (__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i) __builtin_ia32_cvttps2udq128_mask ((__v4sf) __A,
                  (__v4si) __W,
                  (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvttps_epu32 (__mmask8 __U, __m128 __A) {
  return (__m128i) __builtin_ia32_cvttps2udq128_mask ((__v4sf) __A,
                  (__v4si)
                  _mm_setzero_si128 (),
                  (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_cvttps_epu32 (__m256 __A) {
  return (__m256i) __builtin_ia32_cvttps2udq256_mask ((__v8sf) __A,
                  (__v8si)
                  _mm256_setzero_si256 (),
                  (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvttps_epu32 (__m256i __W, __mmask8 __U, __m256 __A) {
  return (__m256i) __builtin_ia32_cvttps2udq256_mask ((__v8sf) __A,
                  (__v8si) __W,
                  (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvttps_epu32 (__mmask8 __U, __m256 __A) {
  return (__m256i) __builtin_ia32_cvttps2udq256_mask ((__v8sf) __A,
                  (__v8si)
                  _mm256_setzero_si256 (),
                  (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cvtepu32_pd (__m128i __A) {
  return (__m128d) __builtin_ia32_cvtudq2pd128_mask ((__v4si) __A,
                 (__v2df)
                 _mm_setzero_pd (),
                 (__mmask8) -1);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_cvtepu32_pd (__m128d __W, __mmask8 __U, __m128i __A) {
  return (__m128d) __builtin_ia32_cvtudq2pd128_mask ((__v4si) __A,
                 (__v2df) __W,
                 (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_cvtepu32_pd (__mmask8 __U, __m128i __A) {
  return (__m128d) __builtin_ia32_cvtudq2pd128_mask ((__v4si) __A,
                 (__v2df)
                 _mm_setzero_pd (),
                 (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_cvtepu32_pd (__m128i __A) {
  return (__m256d) __builtin_ia32_cvtudq2pd256_mask ((__v4si) __A,
                 (__v4df)
                 _mm256_setzero_pd (),
                 (__mmask8) -1);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_cvtepu32_pd (__m256d __W, __mmask8 __U, __m128i __A) {
  return (__m256d) __builtin_ia32_cvtudq2pd256_mask ((__v4si) __A,
                 (__v4df) __W,
                 (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepu32_pd (__mmask8 __U, __m128i __A) {
  return (__m256d) __builtin_ia32_cvtudq2pd256_mask ((__v4si) __A,
                 (__v4df)
                 _mm256_setzero_pd (),
                 (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtepu32_ps (__m128i __A) {
  return (__m128) __builtin_ia32_cvtudq2ps128_mask ((__v4si) __A,
                (__v4sf)
                _mm_setzero_ps (),
                (__mmask8) -1);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_cvtepu32_ps (__m128 __W, __mmask8 __U, __m128i __A) {
  return (__m128) __builtin_ia32_cvtudq2ps128_mask ((__v4si) __A,
                (__v4sf) __W,
                (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_cvtepu32_ps (__mmask8 __U, __m128i __A) {
  return (__m128) __builtin_ia32_cvtudq2ps128_mask ((__v4si) __A,
                (__v4sf)
                _mm_setzero_ps (),
                (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_cvtepu32_ps (__m256i __A) {
  return (__m256) __builtin_ia32_cvtudq2ps256_mask ((__v8si) __A,
                (__v8sf)
                _mm256_setzero_ps (),
                (__mmask8) -1);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_cvtepu32_ps (__m256 __W, __mmask8 __U, __m256i __A) {
  return (__m256) __builtin_ia32_cvtudq2ps256_mask ((__v8si) __A,
                (__v8sf) __W,
                (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepu32_ps (__mmask8 __U, __m256i __A) {
  return (__m256) __builtin_ia32_cvtudq2ps256_mask ((__v8si) __A,
                (__v8sf)
                _mm256_setzero_ps (),
                (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_div_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_divpd_mask ((__v2df) __A,
                (__v2df) __B,
                (__v2df) __W,
                (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_div_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_divpd_mask ((__v2df) __A,
                (__v2df) __B,
                (__v2df)
                _mm_setzero_pd (),
                (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_div_pd (__m256d __W, __mmask8 __U, __m256d __A,
        __m256d __B) {
  return (__m256d) __builtin_ia32_divpd256_mask ((__v4df) __A,
             (__v4df) __B,
             (__v4df) __W,
             (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_div_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  return (__m256d) __builtin_ia32_divpd256_mask ((__v4df) __A,
             (__v4df) __B,
             (__v4df)
             _mm256_setzero_pd (),
             (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_div_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_divps_mask ((__v4sf) __A,
               (__v4sf) __B,
               (__v4sf) __W,
               (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_div_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_divps_mask ((__v4sf) __A,
               (__v4sf) __B,
               (__v4sf)
               _mm_setzero_ps (),
               (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_div_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_divps256_mask ((__v8sf) __A,
            (__v8sf) __B,
            (__v8sf) __W,
            (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_div_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_divps256_mask ((__v8sf) __A,
            (__v8sf) __B,
            (__v8sf)
            _mm256_setzero_ps (),
            (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_expand_pd (__m128d __W, __mmask8 __U, __m128d __A) {
  return (__m128d) __builtin_ia32_expanddf128_mask ((__v2df) __A,
                (__v2df) __W,
                (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_expand_pd (__mmask8 __U, __m128d __A) {
  return (__m128d) __builtin_ia32_expanddf128_mask ((__v2df) __A,
                 (__v2df)
                 _mm_setzero_pd (),
                 (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_expand_pd (__m256d __W, __mmask8 __U, __m256d __A) {
  return (__m256d) __builtin_ia32_expanddf256_mask ((__v4df) __A,
                (__v4df) __W,
                (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_expand_pd (__mmask8 __U, __m256d __A) {
  return (__m256d) __builtin_ia32_expanddf256_mask ((__v4df) __A,
                 (__v4df)
                 _mm256_setzero_pd (),
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_expand_epi64 (__m128i __W, __mmask8 __U, __m128i __A) {
  return (__m128i) __builtin_ia32_expanddi128_mask ((__v2di) __A,
                (__v2di) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_expand_epi64 (__mmask8 __U, __m128i __A) {
  return (__m128i) __builtin_ia32_expanddi128_mask ((__v2di) __A,
                 (__v2di)
                 _mm_setzero_si128 (),
                 (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_expand_epi64 (__m256i __W, __mmask8 __U, __m256i __A) {
  return (__m256i) __builtin_ia32_expanddi256_mask ((__v4di) __A,
                (__v4di) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_expand_epi64 (__mmask8 __U, __m256i __A) {
  return (__m256i) __builtin_ia32_expanddi256_mask ((__v4di) __A,
                 (__v4di)
                 _mm256_setzero_si256 (),
                 (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_expandloadu_pd (__m128d __W, __mmask8 __U, void const *__P) {
  return (__m128d) __builtin_ia32_expandloaddf128_mask ((__v2df *) __P,
              (__v2df) __W,
              (__mmask8)
              __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_expandloadu_pd (__mmask8 __U, void const *__P) {
  return (__m128d) __builtin_ia32_expandloaddf128_mask ((__v2df *) __P,
               (__v2df)
               _mm_setzero_pd (),
               (__mmask8)
               __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_expandloadu_pd (__m256d __W, __mmask8 __U, void const *__P) {
  return (__m256d) __builtin_ia32_expandloaddf256_mask ((__v4df *) __P,
              (__v4df) __W,
              (__mmask8)
              __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_expandloadu_pd (__mmask8 __U, void const *__P) {
  return (__m256d) __builtin_ia32_expandloaddf256_mask ((__v4df *) __P,
               (__v4df)
               _mm256_setzero_pd (),
               (__mmask8)
               __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_expandloadu_epi64 (__m128i __W, __mmask8 __U, void const *__P) {
  return (__m128i) __builtin_ia32_expandloaddi128_mask ((__v2di *) __P,
              (__v2di) __W,
              (__mmask8)
              __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_expandloadu_epi64 (__mmask8 __U, void const *__P) {
  return (__m128i) __builtin_ia32_expandloaddi128_mask ((__v2di *) __P,
               (__v2di)
               _mm_setzero_si128 (),
               (__mmask8)
               __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_expandloadu_epi64 (__m256i __W, __mmask8 __U,
             void const *__P) {
  return (__m256i) __builtin_ia32_expandloaddi256_mask ((__v4di *) __P,
              (__v4di) __W,
              (__mmask8)
              __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_expandloadu_epi64 (__mmask8 __U, void const *__P) {
  return (__m256i) __builtin_ia32_expandloaddi256_mask ((__v4di *) __P,
               (__v4di)
               _mm256_setzero_si256 (),
               (__mmask8)
               __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_expandloadu_ps (__m128 __W, __mmask8 __U, void const *__P) {
  return (__m128) __builtin_ia32_expandloadsf128_mask ((__v4sf *) __P,
                   (__v4sf) __W,
                   (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_expandloadu_ps (__mmask8 __U, void const *__P) {
  return (__m128) __builtin_ia32_expandloadsf128_mask ((__v4sf *) __P,
              (__v4sf)
              _mm_setzero_ps (),
              (__mmask8)
              __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_expandloadu_ps (__m256 __W, __mmask8 __U, void const *__P) {
  return (__m256) __builtin_ia32_expandloadsf256_mask ((__v8sf *) __P,
                   (__v8sf) __W,
                   (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_expandloadu_ps (__mmask8 __U, void const *__P) {
  return (__m256) __builtin_ia32_expandloadsf256_mask ((__v8sf *) __P,
              (__v8sf)
              _mm256_setzero_ps (),
              (__mmask8)
              __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_expandloadu_epi32 (__m128i __W, __mmask8 __U, void const *__P) {
  return (__m128i) __builtin_ia32_expandloadsi128_mask ((__v4si *) __P,
              (__v4si) __W,
              (__mmask8)
              __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_expandloadu_epi32 (__mmask8 __U, void const *__P) {
  return (__m128i) __builtin_ia32_expandloadsi128_mask ((__v4si *) __P,
               (__v4si)
               _mm_setzero_si128 (),
               (__mmask8)     __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_expandloadu_epi32 (__m256i __W, __mmask8 __U,
             void const *__P) {
  return (__m256i) __builtin_ia32_expandloadsi256_mask ((__v8si *) __P,
              (__v8si) __W,
              (__mmask8)
              __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_expandloadu_epi32 (__mmask8 __U, void const *__P) {
  return (__m256i) __builtin_ia32_expandloadsi256_mask ((__v8si *) __P,
               (__v8si)
               _mm256_setzero_si256 (),
               (__mmask8)
               __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_expand_ps (__m128 __W, __mmask8 __U, __m128 __A) {
  return (__m128) __builtin_ia32_expandsf128_mask ((__v4sf) __A,
               (__v4sf) __W,
               (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_expand_ps (__mmask8 __U, __m128 __A) {
  return (__m128) __builtin_ia32_expandsf128_mask ((__v4sf) __A,
                (__v4sf)
                _mm_setzero_ps (),
                (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_expand_ps (__m256 __W, __mmask8 __U, __m256 __A) {
  return (__m256) __builtin_ia32_expandsf256_mask ((__v8sf) __A,
               (__v8sf) __W,
               (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_expand_ps (__mmask8 __U, __m256 __A) {
  return (__m256) __builtin_ia32_expandsf256_mask ((__v8sf) __A,
                (__v8sf)
                _mm256_setzero_ps (),
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_expand_epi32 (__m128i __W, __mmask8 __U, __m128i __A) {
  return (__m128i) __builtin_ia32_expandsi128_mask ((__v4si) __A,
                (__v4si) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_expand_epi32 (__mmask8 __U, __m128i __A) {
  return (__m128i) __builtin_ia32_expandsi128_mask ((__v4si) __A,
                 (__v4si)
                 _mm_setzero_si128 (),
                 (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_expand_epi32 (__m256i __W, __mmask8 __U, __m256i __A) {
  return (__m256i) __builtin_ia32_expandsi256_mask ((__v8si) __A,
                (__v8si) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_expand_epi32 (__mmask8 __U, __m256i __A) {
  return (__m256i) __builtin_ia32_expandsi256_mask ((__v8si) __A,
                 (__v8si)
                 _mm256_setzero_si256 (),
                 (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_getexp_pd (__m128d __A) {
  return (__m128d) __builtin_ia32_getexppd128_mask ((__v2df) __A,
                (__v2df)
                _mm_setzero_pd (),
                (__mmask8) -1);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_getexp_pd (__m128d __W, __mmask8 __U, __m128d __A) {
  return (__m128d) __builtin_ia32_getexppd128_mask ((__v2df) __A,
                (__v2df) __W,
                (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_getexp_pd (__mmask8 __U, __m128d __A) {
  return (__m128d) __builtin_ia32_getexppd128_mask ((__v2df) __A,
                (__v2df)
                _mm_setzero_pd (),
                (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_getexp_pd (__m256d __A) {
  return (__m256d) __builtin_ia32_getexppd256_mask ((__v4df) __A,
                (__v4df)
                _mm256_setzero_pd (),
                (__mmask8) -1);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_getexp_pd (__m256d __W, __mmask8 __U, __m256d __A) {
  return (__m256d) __builtin_ia32_getexppd256_mask ((__v4df) __A,
                (__v4df) __W,
                (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_getexp_pd (__mmask8 __U, __m256d __A) {
  return (__m256d) __builtin_ia32_getexppd256_mask ((__v4df) __A,
                (__v4df)
                _mm256_setzero_pd (),
                (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_getexp_ps (__m128 __A) {
  return (__m128) __builtin_ia32_getexpps128_mask ((__v4sf) __A,
               (__v4sf)
               _mm_setzero_ps (),
               (__mmask8) -1);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_getexp_ps (__m128 __W, __mmask8 __U, __m128 __A) {
  return (__m128) __builtin_ia32_getexpps128_mask ((__v4sf) __A,
               (__v4sf) __W,
               (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_getexp_ps (__mmask8 __U, __m128 __A) {
  return (__m128) __builtin_ia32_getexpps128_mask ((__v4sf) __A,
               (__v4sf)
               _mm_setzero_ps (),
               (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_getexp_ps (__m256 __A) {
  return (__m256) __builtin_ia32_getexpps256_mask ((__v8sf) __A,
               (__v8sf)
               _mm256_setzero_ps (),
               (__mmask8) -1);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_getexp_ps (__m256 __W, __mmask8 __U, __m256 __A) {
  return (__m256) __builtin_ia32_getexpps256_mask ((__v8sf) __A,
               (__v8sf) __W,
               (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_getexp_ps (__mmask8 __U, __m256 __A) {
  return (__m256) __builtin_ia32_getexpps256_mask ((__v8sf) __A,
               (__v8sf)
               _mm256_setzero_ps (),
               (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_max_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_maxpd_mask ((__v2df) __A,
                (__v2df) __B,
                (__v2df) __W,
                (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_max_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_maxpd_mask ((__v2df) __A,
                (__v2df) __B,
                (__v2df)
                _mm_setzero_pd (),
                (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_max_pd (__m256d __W, __mmask8 __U, __m256d __A,
        __m256d __B) {
  return (__m256d) __builtin_ia32_maxpd256_mask ((__v4df) __A,
             (__v4df) __B,
             (__v4df) __W,
             (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_max_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  return (__m256d) __builtin_ia32_maxpd256_mask ((__v4df) __A,
             (__v4df) __B,
             (__v4df)
             _mm256_setzero_pd (),
             (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_max_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_maxps_mask ((__v4sf) __A,
               (__v4sf) __B,
               (__v4sf) __W,
               (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_max_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_maxps_mask ((__v4sf) __A,
               (__v4sf) __B,
               (__v4sf)
               _mm_setzero_ps (),
               (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_max_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_maxps256_mask ((__v8sf) __A,
            (__v8sf) __B,
            (__v8sf) __W,
            (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_max_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_maxps256_mask ((__v8sf) __A,
            (__v8sf) __B,
            (__v8sf)
            _mm256_setzero_ps (),
            (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_min_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_minpd_mask ((__v2df) __A,
                (__v2df) __B,
                (__v2df) __W,
                (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_min_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_minpd_mask ((__v2df) __A,
                (__v2df) __B,
                (__v2df)
                _mm_setzero_pd (),
                (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_min_pd (__m256d __W, __mmask8 __U, __m256d __A,
        __m256d __B) {
  return (__m256d) __builtin_ia32_minpd256_mask ((__v4df) __A,
             (__v4df) __B,
             (__v4df) __W,
             (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_min_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  return (__m256d) __builtin_ia32_minpd256_mask ((__v4df) __A,
             (__v4df) __B,
             (__v4df)
             _mm256_setzero_pd (),
             (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_min_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_minps_mask ((__v4sf) __A,
               (__v4sf) __B,
               (__v4sf) __W,
               (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_min_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_minps_mask ((__v4sf) __A,
               (__v4sf) __B,
               (__v4sf)
               _mm_setzero_ps (),
               (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_min_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_minps256_mask ((__v8sf) __A,
            (__v8sf) __B,
            (__v8sf) __W,
            (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_min_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_minps256_mask ((__v8sf) __A,
            (__v8sf) __B,
            (__v8sf)
            _mm256_setzero_ps (),
            (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_mul_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_mulpd_mask ((__v2df) __A,
                (__v2df) __B,
                (__v2df) __W,
                (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_mul_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_mulpd_mask ((__v2df) __A,
                (__v2df) __B,
                (__v2df)
                _mm_setzero_pd (),
                (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_mul_pd (__m256d __W, __mmask8 __U, __m256d __A,
        __m256d __B) {
  return (__m256d) __builtin_ia32_mulpd256_mask ((__v4df) __A,
             (__v4df) __B,
             (__v4df) __W,
             (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_mul_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  return (__m256d) __builtin_ia32_mulpd256_mask ((__v4df) __A,
             (__v4df) __B,
             (__v4df)
             _mm256_setzero_pd (),
             (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_mul_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_mulps_mask ((__v4sf) __A,
               (__v4sf) __B,
               (__v4sf) __W,
               (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_mul_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_mulps_mask ((__v4sf) __A,
               (__v4sf) __B,
               (__v4sf)
               _mm_setzero_ps (),
               (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_mul_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_mulps256_mask ((__v8sf) __A,
            (__v8sf) __B,
            (__v8sf) __W,
            (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_mul_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_mulps256_mask ((__v8sf) __A,
            (__v8sf) __B,
            (__v8sf)
            _mm256_setzero_ps (),
            (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_abs_epi32 (__m128i __W, __mmask8 __U, __m128i __A) {
  return (__m128i) __builtin_ia32_pabsd128_mask ((__v4si) __A,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_abs_epi32 (__mmask8 __U, __m128i __A) {
  return (__m128i) __builtin_ia32_pabsd128_mask ((__v4si) __A,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_abs_epi32 (__m256i __W, __mmask8 __U, __m256i __A) {
  return (__m256i) __builtin_ia32_pabsd256_mask ((__v8si) __A,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_abs_epi32 (__mmask8 __U, __m256i __A) {
  return (__m256i) __builtin_ia32_pabsd256_mask ((__v8si) __A,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_abs_epi64 (__m128i __A) {
  return (__m128i) __builtin_ia32_pabsq128_mask ((__v2di) __A,
             (__v2di)
             _mm_setzero_si128 (),
             (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_abs_epi64 (__m128i __W, __mmask8 __U, __m128i __A) {
  return (__m128i) __builtin_ia32_pabsq128_mask ((__v2di) __A,
             (__v2di) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_abs_epi64 (__mmask8 __U, __m128i __A) {
  return (__m128i) __builtin_ia32_pabsq128_mask ((__v2di) __A,
             (__v2di)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_abs_epi64 (__m256i __A) {
  return (__m256i) __builtin_ia32_pabsq256_mask ((__v4di) __A,
             (__v4di)
             _mm256_setzero_si256 (),
             (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_abs_epi64 (__m256i __W, __mmask8 __U, __m256i __A) {
  return (__m256i) __builtin_ia32_pabsq256_mask ((__v4di) __A,
             (__v4di) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_abs_epi64 (__mmask8 __U, __m256i __A) {
  return (__m256i) __builtin_ia32_pabsq256_mask ((__v4di) __A,
             (__v4di)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_max_epi32 (__mmask8 __M, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pmaxsd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si)
              _mm_setzero_si128 (),
              __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_max_epi32 (__m128i __W, __mmask8 __M, __m128i __A,
        __m128i __B) {
  return (__m128i) __builtin_ia32_pmaxsd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si) __W, __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_max_epi32 (__mmask8 __M, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pmaxsd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si)
              _mm256_setzero_si256 (),
              __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_max_epi32 (__m256i __W, __mmask8 __M, __m256i __A,
           __m256i __B) {
  return (__m256i) __builtin_ia32_pmaxsd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si) __W, __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_max_epi64 (__mmask8 __M, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pmaxsq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_si128 (),
              __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_max_epi64 (__m128i __W, __mmask8 __M, __m128i __A,
        __m128i __B) {
  return (__m128i) __builtin_ia32_pmaxsq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di) __W, __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_max_epi64 (__m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pmaxsq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_si128 (),
              (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_max_epi64 (__mmask8 __M, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pmaxsq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_si256 (),
              __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_max_epi64 (__m256i __W, __mmask8 __M, __m256i __A,
           __m256i __B) {
  return (__m256i) __builtin_ia32_pmaxsq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di) __W, __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_max_epi64 (__m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pmaxsq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_si256 (),
              (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_max_epu32 (__mmask8 __M, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pmaxud128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si)
              _mm_setzero_si128 (),
              __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_max_epu32 (__m128i __W, __mmask8 __M, __m128i __A,
        __m128i __B) {
  return (__m128i) __builtin_ia32_pmaxud128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si) __W, __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_max_epu32 (__mmask8 __M, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pmaxud256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si)
              _mm256_setzero_si256 (),
              __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_max_epu32 (__m256i __W, __mmask8 __M, __m256i __A,
           __m256i __B) {
  return (__m256i) __builtin_ia32_pmaxud256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si) __W, __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_max_epu64 (__mmask8 __M, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pmaxuq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_si128 (),
              __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_max_epu64 (__m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pmaxuq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_si128 (),
              (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_max_epu64 (__m128i __W, __mmask8 __M, __m128i __A,
        __m128i __B) {
  return (__m128i) __builtin_ia32_pmaxuq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di) __W, __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_max_epu64 (__mmask8 __M, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pmaxuq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_si256 (),
              __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_max_epu64 (__m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pmaxuq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_si256 (),
              (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_max_epu64 (__m256i __W, __mmask8 __M, __m256i __A,
           __m256i __B) {
  return (__m256i) __builtin_ia32_pmaxuq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di) __W, __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_min_epi32 (__mmask8 __M, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pminsd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si)
              _mm_setzero_si128 (),
              __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_min_epi32 (__m128i __W, __mmask8 __M, __m128i __A,
        __m128i __B) {
  return (__m128i) __builtin_ia32_pminsd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si) __W, __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_min_epi32 (__mmask8 __M, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pminsd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si)
              _mm256_setzero_si256 (),
              __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_min_epi32 (__m256i __W, __mmask8 __M, __m256i __A,
           __m256i __B) {
  return (__m256i) __builtin_ia32_pminsd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si) __W, __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_min_epi64 (__m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pminsq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_si128 (),
              (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_min_epi64 (__m128i __W, __mmask8 __M, __m128i __A,
        __m128i __B) {
  return (__m128i) __builtin_ia32_pminsq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di) __W, __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_min_epi64 (__mmask8 __M, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pminsq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_si128 (),
              __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_min_epi64 (__m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pminsq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_si256 (),
              (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_min_epi64 (__m256i __W, __mmask8 __M, __m256i __A,
           __m256i __B) {
  return (__m256i) __builtin_ia32_pminsq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di) __W, __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_min_epi64 (__mmask8 __M, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pminsq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_si256 (),
              __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_min_epu32 (__mmask8 __M, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pminud128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si)
              _mm_setzero_si128 (),
              __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_min_epu32 (__m128i __W, __mmask8 __M, __m128i __A,
        __m128i __B) {
  return (__m128i) __builtin_ia32_pminud128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si) __W, __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_min_epu32 (__mmask8 __M, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pminud256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si)
              _mm256_setzero_si256 (),
              __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_min_epu32 (__m256i __W, __mmask8 __M, __m256i __A,
           __m256i __B) {
  return (__m256i) __builtin_ia32_pminud256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si) __W, __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_min_epu64 (__m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pminuq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_si128 (),
              (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_min_epu64 (__m128i __W, __mmask8 __M, __m128i __A,
        __m128i __B) {
  return (__m128i) __builtin_ia32_pminuq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di) __W, __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_min_epu64 (__mmask8 __M, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pminuq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_si128 (),
              __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_min_epu64 (__m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pminuq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_si256 (),
              (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_min_epu64 (__m256i __W, __mmask8 __M, __m256i __A,
           __m256i __B) {
  return (__m256i) __builtin_ia32_pminuq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di) __W, __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_min_epu64 (__mmask8 __M, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pminuq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_si256 (),
              __M);
}

#define _mm_roundscale_pd(__A, __imm) __extension__ ({ \
  (__m128d) __builtin_ia32_rndscalepd_128_mask ((__v2df) __A, \
                   __imm, (__v2df) _mm_setzero_pd (), (__mmask8) -1); })


#define _mm_mask_roundscale_pd(__W, __U, __A, __imm) __extension__ ({ \
  (__m128d) __builtin_ia32_rndscalepd_128_mask ((__v2df) __A, __imm, \
                   (__v2df) __W, (__mmask8) __U); })


#define _mm_maskz_roundscale_pd(__U, __A, __imm) __extension__ ({ \
  (__m128d) __builtin_ia32_rndscalepd_128_mask ((__v2df) __A, __imm, \
                   (__v2df) _mm_setzero_pd (), (__mmask8) __U); })


#define _mm256_roundscale_pd(__A, __imm) __extension__ ({ \
  (__m256d) __builtin_ia32_rndscalepd_256_mask ((__v4df) __A, __imm, \
                   (__v4df) _mm256_setzero_pd (), (__mmask8) -1); })


#define _mm256_mask_roundscale_pd(__W, __U, __A, __imm) __extension__ ({ \
  (__m256d) __builtin_ia32_rndscalepd_256_mask ((__v4df) __A, __imm, \
                   (__v4df) __W, (__mmask8) __U); })


#define _mm256_maskz_roundscale_pd(__U, __A, __imm)  __extension__ ({ \
  (__m256d) __builtin_ia32_rndscalepd_256_mask ((__v4df) __A, __imm, \
                   (__v4df) _mm256_setzero_pd(), (__mmask8) __U); })

#define _mm_roundscale_ps(__A, __imm)  __extension__ ({ \
  (__m128) __builtin_ia32_rndscaleps_128_mask ((__v4sf) __A, __imm, \
                  (__v4sf) _mm_setzero_ps(), (__mmask8) -1); })


#define _mm_mask_roundscale_ps(__W, __U, __A, __imm)  __extension__ ({ \
  (__m128) __builtin_ia32_rndscaleps_128_mask ((__v4sf) __A, __imm, \
                  (__v4sf) __W, (__mmask8) __U); })


#define _mm_maskz_roundscale_ps(__U, __A, __imm)  __extension__ ({ \
  (__m128) __builtin_ia32_rndscaleps_128_mask ((__v4sf) __A, __imm, \
                  (__v4sf) _mm_setzero_ps(), (__mmask8) __U); })

#define _mm256_roundscale_ps(__A, __imm)  __extension__ ({ \
  (__m256) __builtin_ia32_rndscaleps_256_mask ((__v8sf) __A,__imm, \
                  (__v8sf) _mm256_setzero_ps(), (__mmask8) -1); })

#define _mm256_mask_roundscale_ps(__W, __U, __A,__imm)  __extension__ ({ \
  (__m256) __builtin_ia32_rndscaleps_256_mask ((__v8sf) __A, __imm, \
                  (__v8sf) __W, (__mmask8) __U); })


#define _mm256_maskz_roundscale_ps(__U, __A, __imm)  __extension__ ({ \
  (__m256) __builtin_ia32_rndscaleps_256_mask ((__v8sf) __A, __imm, \
                  (__v8sf) _mm256_setzero_ps(), (__mmask8) __U); })

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_scalef_pd (__m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_scalefpd128_mask ((__v2df) __A,
                (__v2df) __B,
                (__v2df)
                _mm_setzero_pd (),
                (__mmask8) -1);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_scalef_pd (__m128d __W, __mmask8 __U, __m128d __A,
        __m128d __B) {
  return (__m128d) __builtin_ia32_scalefpd128_mask ((__v2df) __A,
                (__v2df) __B,
                (__v2df) __W,
                (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_scalef_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_scalefpd128_mask ((__v2df) __A,
                (__v2df) __B,
                (__v2df)
                _mm_setzero_pd (),
                (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_scalef_pd (__m256d __A, __m256d __B) {
  return (__m256d) __builtin_ia32_scalefpd256_mask ((__v4df) __A,
                (__v4df) __B,
                (__v4df)
                _mm256_setzero_pd (),
                (__mmask8) -1);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_scalef_pd (__m256d __W, __mmask8 __U, __m256d __A,
           __m256d __B) {
  return (__m256d) __builtin_ia32_scalefpd256_mask ((__v4df) __A,
                (__v4df) __B,
                (__v4df) __W,
                (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_scalef_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  return (__m256d) __builtin_ia32_scalefpd256_mask ((__v4df) __A,
                (__v4df) __B,
                (__v4df)
                _mm256_setzero_pd (),
                (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_scalef_ps (__m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_scalefps128_mask ((__v4sf) __A,
               (__v4sf) __B,
               (__v4sf)
               _mm_setzero_ps (),
               (__mmask8) -1);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_scalef_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_scalefps128_mask ((__v4sf) __A,
               (__v4sf) __B,
               (__v4sf) __W,
               (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_scalef_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_scalefps128_mask ((__v4sf) __A,
               (__v4sf) __B,
               (__v4sf)
               _mm_setzero_ps (),
               (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_scalef_ps (__m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_scalefps256_mask ((__v8sf) __A,
               (__v8sf) __B,
               (__v8sf)
               _mm256_setzero_ps (),
               (__mmask8) -1);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_scalef_ps (__m256 __W, __mmask8 __U, __m256 __A,
           __m256 __B) {
  return (__m256) __builtin_ia32_scalefps256_mask ((__v8sf) __A,
               (__v8sf) __B,
               (__v8sf) __W,
               (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_scalef_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_scalefps256_mask ((__v8sf) __A,
               (__v8sf) __B,
               (__v8sf)
               _mm256_setzero_ps (),
               (__mmask8) __U);
}

#define _mm_i64scatter_pd(__addr,__index, __v1, __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv2df(__addr, (__mmask8) 0xFF, (__v2di) __index, \
                              (__v2df) __v1, __scale); })

#define _mm_mask_i64scatter_pd(__addr, __mask, __index, __v1, \
                               __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv2df (__addr, __mask, (__v2di) __index, \
                               (__v2df) __v1, __scale); })


#define _mm_i64scatter_epi64(__addr, __index, __v1, __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv2di (__addr, (__mmask8) 0xFF, \
        (__v2di) __index, (__v2di) __v1, __scale); })

#define _mm_mask_i64scatter_epi64(__addr, __mask, __index, __v1,\
                                  __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv2di (__addr, __mask, (__v2di) __index,\
        (__v2di) __v1, __scale); })

#define _mm256_i64scatter_pd(__addr, __index, __v1, __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv4df (__addr, (__mmask8) 0xFF,\
        (__v4di) __index, (__v4df) __v1, __scale); })

#define _mm256_mask_i64scatter_pd(__addr, __mask, __index, __v1,\
                                   __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv4df (__addr, __mask, (__v4di) __index,\
        (__v4df) __v1, __scale); })

#define _mm256_i64scatter_epi64(__addr, __index, __v1, __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv4di (__addr, (__mmask8) 0xFF, (__v4di) __index,\
                               (__v4di) __v1, __scale); })

#define _mm256_mask_i64scatter_epi64(__addr, __mask, __index, __v1,\
                                      __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv4di (__addr, __mask, (__v4di) __index,\
        (__v4di) __v1, __scale); })

#define _mm_i64scatter_ps(__addr, __index, __v1, __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv4sf (__addr, (__mmask8) 0xFF,\
        (__v2di) __index, (__v4sf) __v1, __scale); })

#define _mm_mask_i64scatter_ps(__addr, __mask, __index, __v1, \
                                __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv4sf (__addr, __mask, (__v2di) __index,\
        (__v4sf) __v1, __scale); })

#define _mm_i64scatter_epi32(__addr, __index, __v1, \
                              __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv4si (__addr, (__mmask8) 0xFF,\
        (__v2di) __index, (__v4si) __v1, __scale); })

#define _mm_mask_i64scatter_epi32(__addr, __mask, __index, __v1,\
         __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv4si (__addr, __mask, (__v2di) __index,\
        (__v4si) __v1, __scale); })

#define _mm256_i64scatter_ps(__addr, __index, __v1, __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv8sf (__addr, (__mmask8) 0xFF, (__v4di) __index, \
                              (__v4sf) __v1, __scale); })

#define _mm256_mask_i64scatter_ps(__addr, __mask, __index, __v1, \
                                   __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv8sf (__addr, __mask, (__v4di) __index, \
        (__v4sf) __v1, __scale); })

#define _mm256_i64scatter_epi32(__addr, __index, __v1, __scale) __extension__ ({ \
  __builtin_ia32_scatterdiv8si (__addr, (__mmask8) 0xFF, \
        (__v4di) __index, (__v4si) __v1, __scale); })

#define _mm256_mask_i64scatter_epi32(__addr, __mask, __index, __v1, \
                                      __scale) __extension__ ({  \
  __builtin_ia32_scatterdiv8si(__addr, __mask, (__v4di) __index, \
        (__v4si) __v1, __scale); })

#define _mm_i32scatter_pd(__addr, __index, __v1,         \
                          __scale) __extension__ ({      \
  __builtin_ia32_scattersiv2df (__addr, (__mmask8) 0xFF, \
        (__v4si) __index, (__v2df) __v1, __scale); })

#define _mm_mask_i32scatter_pd(__addr, __mask, __index, __v1,    \
                                __scale) __extension__ ({        \
  __builtin_ia32_scattersiv2df (__addr, __mask, (__v4si) __index,\
         (__v2df) __v1, __scale); })

#define _mm_i32scatter_epi64(__addr, __index, __v1, __scale) __extension__ ({ \
  __builtin_ia32_scattersiv2di (__addr, (__mmask8) 0xFF,                       \
        (__v4si) __index, (__v2di) __v1, __scale); })

#define _mm_mask_i32scatter_epi64(__addr, __mask, __index, __v1, \
         __scale) __extension__ ({                                \
  __builtin_ia32_scattersiv2di (__addr, __mask, (__v4si) __index, \
        (__v2di) __v1, __scale); })

#define _mm256_i32scatter_pd(__addr, __index, __v1, __scale) __extension__ ({ \
  __builtin_ia32_scattersiv4df (__addr, (__mmask8) 0xFF,                      \
        (__v4si) __index, (__v4df) __v1, __scale); })

#define _mm256_mask_i32scatter_pd(__addr, __mask, __index, __v1, \
         __scale) __extension__ ({                                \
  __builtin_ia32_scattersiv4df (__addr, __mask, (__v4si) __index, \
        (__v4df) __v1, __scale); })

#define _mm256_i32scatter_epi64(__addr, __index, __v1,    \
                                __scale) __extension__ ({ \
  __builtin_ia32_scattersiv4di (__addr, (__mmask8) 0xFF,  \
        (__v4si) __index, (__v4di) __v1, __scale); })

#define _mm256_mask_i32scatter_epi64(__addr, __mask, __index, __v1, \
            __scale) __extension__ ({                               \
  __builtin_ia32_scattersiv4di (__addr, __mask, (__v4si) __index,   \
        (__v4di) __v1, __scale); })

#define _mm_i32scatter_ps(__addr, __index, __v1, __scale) __extension__ ({ \
  __builtin_ia32_scattersiv4sf (__addr, (__mmask8) 0xFF,                   \
        (__v4si) __index, (__v4sf) __v1, __scale); })

#define _mm_mask_i32scatter_ps(__addr, __mask, __index, __v1,     \
                               __scale) __extension__ ({          \
  __builtin_ia32_scattersiv4sf (__addr, __mask, (__v4si) __index, \
        (__v4sf) __v1, __scale); })

#define _mm_i32scatter_epi32(__addr, __index, __v1, __scale) __extension__ ({ \
  __builtin_ia32_scattersiv4si (__addr, (__mmask8) 0xFF,                       \
        (__v4si) __index, (__v4si) __v1, __scale); })

#define _mm_mask_i32scatter_epi32(__addr, __mask, __index, __v1, \
                                  __scale) __extension__ ({      \
  __builtin_ia32_scattersiv4si (__addr, __mask, (__v4si) __index,\
        (__v4si) __v1, __scale); })

#define _mm256_i32scatter_ps(__addr, __index, __v1, __scale) __extension__ ({ \
  __builtin_ia32_scattersiv8sf (__addr, (__mmask8) 0xFF,                      \
        (__v8si) __index, (__v8sf) __v1, __scale); })

#define _mm256_mask_i32scatter_ps(__addr, __mask, __index, __v1, \
                                   __scale) __extension__ ({     \
  __builtin_ia32_scattersiv8sf (__addr, __mask, (__v8si) __index,\
        (__v8sf) __v1, __scale); })

#define _mm256_i32scatter_epi32(__addr, __index, __v1, __scale) __extension__ ({ \
  __builtin_ia32_scattersiv8si (__addr, (__mmask8) 0xFF,                         \
        (__v8si) __index, (__v8si) __v1, __scale); })

#define _mm256_mask_i32scatter_epi32(__addr, __mask, __index, __v1, \
            __scale) __extension__ ({                                \
  __builtin_ia32_scattersiv8si (__addr, __mask, (__v8si) __index,    \
        (__v8si) __v1, __scale); })

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_sqrt_pd (__m128d __W, __mmask8 __U, __m128d __A) {
  return (__m128d) __builtin_ia32_sqrtpd128_mask ((__v2df) __A,
              (__v2df) __W,
              (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_sqrt_pd (__mmask8 __U, __m128d __A) {
  return (__m128d) __builtin_ia32_sqrtpd128_mask ((__v2df) __A,
              (__v2df)
              _mm_setzero_pd (),
              (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_sqrt_pd (__m256d __W, __mmask8 __U, __m256d __A) {
  return (__m256d) __builtin_ia32_sqrtpd256_mask ((__v4df) __A,
              (__v4df) __W,
              (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_sqrt_pd (__mmask8 __U, __m256d __A) {
  return (__m256d) __builtin_ia32_sqrtpd256_mask ((__v4df) __A,
              (__v4df)
              _mm256_setzero_pd (),
              (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_sqrt_ps (__m128 __W, __mmask8 __U, __m128 __A) {
  return (__m128) __builtin_ia32_sqrtps128_mask ((__v4sf) __A,
             (__v4sf) __W,
             (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_sqrt_ps (__mmask8 __U, __m128 __A) {
  return (__m128) __builtin_ia32_sqrtps128_mask ((__v4sf) __A,
             (__v4sf)
             _mm_setzero_ps (),
             (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_sqrt_ps (__m256 __W, __mmask8 __U, __m256 __A) {
  return (__m256) __builtin_ia32_sqrtps256_mask ((__v8sf) __A,
             (__v8sf) __W,
             (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_sqrt_ps (__mmask8 __U, __m256 __A) {
  return (__m256) __builtin_ia32_sqrtps256_mask ((__v8sf) __A,
             (__v8sf)
             _mm256_setzero_ps (),
             (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_sub_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_subpd128_mask ((__v2df) __A,
             (__v2df) __B,
             (__v2df) __W,
             (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_sub_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  return (__m128d) __builtin_ia32_subpd128_mask ((__v2df) __A,
             (__v2df) __B,
             (__v2df)
             _mm_setzero_pd (),
             (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_sub_pd (__m256d __W, __mmask8 __U, __m256d __A,
        __m256d __B) {
  return (__m256d) __builtin_ia32_subpd256_mask ((__v4df) __A,
             (__v4df) __B,
             (__v4df) __W,
             (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_sub_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  return (__m256d) __builtin_ia32_subpd256_mask ((__v4df) __A,
             (__v4df) __B,
             (__v4df)
             _mm256_setzero_pd (),
             (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_sub_ps (__m128 __W, __mmask16 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_subps128_mask ((__v4sf) __A,
            (__v4sf) __B,
            (__v4sf) __W,
            (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_sub_ps (__mmask16 __U, __m128 __A, __m128 __B) {
  return (__m128) __builtin_ia32_subps128_mask ((__v4sf) __A,
            (__v4sf) __B,
            (__v4sf)
            _mm_setzero_ps (),
            (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_sub_ps (__m256 __W, __mmask16 __U, __m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_subps256_mask ((__v8sf) __A,
            (__v8sf) __B,
            (__v8sf) __W,
            (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_sub_ps (__mmask16 __U, __m256 __A, __m256 __B) {
  return (__m256) __builtin_ia32_subps256_mask ((__v8sf) __A,
            (__v8sf) __B,
            (__v8sf)
            _mm256_setzero_ps (),
            (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask2_permutex2var_epi32 (__m128i __A, __m128i __I, __mmask8 __U,
            __m128i __B) {
  return (__m128i) __builtin_ia32_vpermi2vard128_mask ((__v4si) __A,
                   (__v4si) __I
                   /* idx */ ,
                   (__v4si) __B,
                   (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask2_permutex2var_epi32 (__m256i __A, __m256i __I,
         __mmask8 __U, __m256i __B) {
  return (__m256i) __builtin_ia32_vpermi2vard256_mask ((__v8si) __A,
                   (__v8si) __I
                   /* idx */ ,
                   (__v8si) __B,
                   (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask2_permutex2var_pd (__m128d __A, __m128i __I, __mmask8 __U,
         __m128d __B) {
  return (__m128d) __builtin_ia32_vpermi2varpd128_mask ((__v2df) __A,
              (__v2di) __I
              /* idx */ ,
              (__v2df) __B,
              (__mmask8)
              __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask2_permutex2var_pd (__m256d __A, __m256i __I, __mmask8 __U,
            __m256d __B) {
  return (__m256d) __builtin_ia32_vpermi2varpd256_mask ((__v4df) __A,
              (__v4di) __I
              /* idx */ ,
              (__v4df) __B,
              (__mmask8)
              __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask2_permutex2var_ps (__m128 __A, __m128i __I, __mmask8 __U,
         __m128 __B) {
  return (__m128) __builtin_ia32_vpermi2varps128_mask ((__v4sf) __A,
                   (__v4si) __I
                   /* idx */ ,
                   (__v4sf) __B,
                   (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask2_permutex2var_ps (__m256 __A, __m256i __I, __mmask8 __U,
            __m256 __B) {
  return (__m256) __builtin_ia32_vpermi2varps256_mask ((__v8sf) __A,
                   (__v8si) __I
                   /* idx */ ,
                   (__v8sf) __B,
                   (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask2_permutex2var_epi64 (__m128i __A, __m128i __I, __mmask8 __U,
            __m128i __B) {
  return (__m128i) __builtin_ia32_vpermi2varq128_mask ((__v2di) __A,
                   (__v2di) __I
                   /* idx */ ,
                   (__v2di) __B,
                   (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask2_permutex2var_epi64 (__m256i __A, __m256i __I,
         __mmask8 __U, __m256i __B) {
  return (__m256i) __builtin_ia32_vpermi2varq256_mask ((__v4di) __A,
                   (__v4di) __I
                   /* idx */ ,
                   (__v4di) __B,
                   (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_permutex2var_epi32 (__m128i __A, __m128i __I, __m128i __B) {
  return (__m128i) __builtin_ia32_vpermt2vard128_mask ((__v4si) __I
                   /* idx */ ,
                   (__v4si) __A,
                   (__v4si) __B,
                   (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_permutex2var_epi32 (__m128i __A, __mmask8 __U, __m128i __I,
           __m128i __B) {
  return (__m128i) __builtin_ia32_vpermt2vard128_mask ((__v4si) __I
                   /* idx */ ,
                   (__v4si) __A,
                   (__v4si) __B,
                   (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_permutex2var_epi32 (__mmask8 __U, __m128i __A, __m128i __I,
            __m128i __B) {
  return (__m128i) __builtin_ia32_vpermt2vard128_maskz ((__v4si) __I
              /* idx */ ,
              (__v4si) __A,
              (__v4si) __B,
              (__mmask8)
              __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_permutex2var_epi32 (__m256i __A, __m256i __I, __m256i __B) {
  return (__m256i) __builtin_ia32_vpermt2vard256_mask ((__v8si) __I
                   /* idx */ ,
                   (__v8si) __A,
                   (__v8si) __B,
                   (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_permutex2var_epi32 (__m256i __A, __mmask8 __U, __m256i __I,
        __m256i __B) {
  return (__m256i) __builtin_ia32_vpermt2vard256_mask ((__v8si) __I
                   /* idx */ ,
                   (__v8si) __A,
                   (__v8si) __B,
                   (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_permutex2var_epi32 (__mmask8 __U, __m256i __A,
         __m256i __I, __m256i __B) {
  return (__m256i) __builtin_ia32_vpermt2vard256_maskz ((__v8si) __I
              /* idx */ ,
              (__v8si) __A,
              (__v8si) __B,
              (__mmask8)
              __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_permutex2var_pd (__m128d __A, __m128i __I, __m128d __B) {
  return (__m128d) __builtin_ia32_vpermt2varpd128_mask ((__v2di) __I
              /* idx */ ,
              (__v2df) __A,
              (__v2df) __B,
              (__mmask8) -
              1);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_permutex2var_pd (__m128d __A, __mmask8 __U, __m128i __I,
        __m128d __B) {
  return (__m128d) __builtin_ia32_vpermt2varpd128_mask ((__v2di) __I
              /* idx */ ,
              (__v2df) __A,
              (__v2df) __B,
              (__mmask8)
              __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_permutex2var_pd (__mmask8 __U, __m128d __A, __m128i __I,
         __m128d __B) {
  return (__m128d) __builtin_ia32_vpermt2varpd128_maskz ((__v2di) __I
               /* idx */ ,
               (__v2df) __A,
               (__v2df) __B,
               (__mmask8)
               __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_permutex2var_pd (__m256d __A, __m256i __I, __m256d __B) {
  return (__m256d) __builtin_ia32_vpermt2varpd256_mask ((__v4di) __I
              /* idx */ ,
              (__v4df) __A,
              (__v4df) __B,
              (__mmask8) -
              1);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_permutex2var_pd (__m256d __A, __mmask8 __U, __m256i __I,
           __m256d __B) {
  return (__m256d) __builtin_ia32_vpermt2varpd256_mask ((__v4di) __I
              /* idx */ ,
              (__v4df) __A,
              (__v4df) __B,
              (__mmask8)
              __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_permutex2var_pd (__mmask8 __U, __m256d __A, __m256i __I,
            __m256d __B) {
  return (__m256d) __builtin_ia32_vpermt2varpd256_maskz ((__v4di) __I
               /* idx */ ,
               (__v4df) __A,
               (__v4df) __B,
               (__mmask8)
               __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_permutex2var_ps (__m128 __A, __m128i __I, __m128 __B) {
  return (__m128) __builtin_ia32_vpermt2varps128_mask ((__v4si) __I
                   /* idx */ ,
                   (__v4sf) __A,
                   (__v4sf) __B,
                   (__mmask8) -1);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mask_permutex2var_ps (__m128 __A, __mmask8 __U, __m128i __I,
        __m128 __B) {
  return (__m128) __builtin_ia32_vpermt2varps128_mask ((__v4si) __I
                   /* idx */ ,
                   (__v4sf) __A,
                   (__v4sf) __B,
                   (__mmask8) __U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_maskz_permutex2var_ps (__mmask8 __U, __m128 __A, __m128i __I,
         __m128 __B) {
  return (__m128) __builtin_ia32_vpermt2varps128_maskz ((__v4si) __I
              /* idx */ ,
              (__v4sf) __A,
              (__v4sf) __B,
              (__mmask8)
              __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_permutex2var_ps (__m256 __A, __m256i __I, __m256 __B) {
  return (__m256) __builtin_ia32_vpermt2varps256_mask ((__v8si) __I
                   /* idx */ ,
                   (__v8sf) __A,
                   (__v8sf) __B,
                   (__mmask8) -1);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_mask_permutex2var_ps (__m256 __A, __mmask8 __U, __m256i __I,
           __m256 __B) {
  return (__m256) __builtin_ia32_vpermt2varps256_mask ((__v8si) __I
                   /* idx */ ,
                   (__v8sf) __A,
                   (__v8sf) __B,
                   (__mmask8) __U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS
_mm256_maskz_permutex2var_ps (__mmask8 __U, __m256 __A, __m256i __I,
            __m256 __B) {
  return (__m256) __builtin_ia32_vpermt2varps256_maskz ((__v8si) __I
              /* idx */ ,
              (__v8sf) __A,
              (__v8sf) __B,
              (__mmask8)
              __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_permutex2var_epi64 (__m128i __A, __m128i __I, __m128i __B) {
  return (__m128i) __builtin_ia32_vpermt2varq128_mask ((__v2di) __I
                   /* idx */ ,
                   (__v2di) __A,
                   (__v2di) __B,
                   (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_permutex2var_epi64 (__m128i __A, __mmask8 __U, __m128i __I,
           __m128i __B) {
  return (__m128i) __builtin_ia32_vpermt2varq128_mask ((__v2di) __I
                   /* idx */ ,
                   (__v2di) __A,
                   (__v2di) __B,
                   (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_permutex2var_epi64 (__mmask8 __U, __m128i __A, __m128i __I,
            __m128i __B) {
  return (__m128i) __builtin_ia32_vpermt2varq128_maskz ((__v2di) __I
              /* idx */ ,
              (__v2di) __A,
              (__v2di) __B,
              (__mmask8)
              __U);
}


static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_permutex2var_epi64 (__m256i __A, __m256i __I, __m256i __B) {
  return (__m256i) __builtin_ia32_vpermt2varq256_mask ((__v4di) __I
                   /* idx */ ,
                   (__v4di) __A,
                   (__v4di) __B,
                   (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_permutex2var_epi64 (__m256i __A, __mmask8 __U, __m256i __I,
        __m256i __B) {
  return (__m256i) __builtin_ia32_vpermt2varq256_mask ((__v4di) __I
                   /* idx */ ,
                   (__v4di) __A,
                   (__v4di) __B,
                   (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_permutex2var_epi64 (__mmask8 __U, __m256i __A,
         __m256i __I, __m256i __B) {
  return (__m256i) __builtin_ia32_vpermt2varq256_maskz ((__v4di) __I
              /* idx */ ,
              (__v4di) __A,
              (__v4di) __B,
              (__mmask8)
              __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtepi8_epi32 (__m128i __W, __mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovsxbd128_mask ((__v16qi) __A,
                (__v4si) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtepi8_epi32 (__mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovsxbd128_mask ((__v16qi) __A,
                (__v4si)
                _mm_setzero_si128 (),
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvtepi8_epi32 (__m256i __W, __mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovsxbd256_mask ((__v16qi) __A,
                (__v8si) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepi8_epi32 (__mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovsxbd256_mask ((__v16qi) __A,
                (__v8si)
                _mm256_setzero_si256 (),
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtepi8_epi64 (__m128i __W, __mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovsxbq128_mask ((__v16qi) __A,
                (__v2di) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtepi8_epi64 (__mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovsxbq128_mask ((__v16qi) __A,
                (__v2di)
                _mm_setzero_si128 (),
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvtepi8_epi64 (__m256i __W, __mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovsxbq256_mask ((__v16qi) __A,
                (__v4di) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepi8_epi64 (__mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovsxbq256_mask ((__v16qi) __A,
                (__v4di)
                _mm256_setzero_si256 (),
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtepi32_epi64 (__m128i __W, __mmask8 __U, __m128i __X)
{
  return (__m128i) __builtin_ia32_pmovsxdq128_mask ((__v4si) __X,
                (__v2di) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtepi32_epi64 (__mmask8 __U, __m128i __X)
{
  return (__m128i) __builtin_ia32_pmovsxdq128_mask ((__v4si) __X,
                (__v2di)
                _mm_setzero_si128 (),
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvtepi32_epi64 (__m256i __W, __mmask8 __U, __m128i __X)
{
  return (__m256i) __builtin_ia32_pmovsxdq256_mask ((__v4si) __X,
                (__v4di) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepi32_epi64 (__mmask8 __U, __m128i __X)
{
  return (__m256i) __builtin_ia32_pmovsxdq256_mask ((__v4si) __X,
                (__v4di)
                _mm256_setzero_si256 (),
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtepi16_epi32 (__m128i __W, __mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovsxwd128_mask ((__v8hi) __A,
                (__v4si) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtepi16_epi32 (__mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovsxwd128_mask ((__v8hi) __A,
                (__v4si)
                _mm_setzero_si128 (),
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvtepi16_epi32 (__m256i __W, __mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovsxwd256_mask ((__v8hi) __A,
                (__v8si) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepi16_epi32 (__mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovsxwd256_mask ((__v8hi) __A,
                (__v8si)
                _mm256_setzero_si256 (),
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtepi16_epi64 (__m128i __W, __mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovsxwq128_mask ((__v8hi) __A,
                (__v2di) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtepi16_epi64 (__mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovsxwq128_mask ((__v8hi) __A,
                (__v2di)
                _mm_setzero_si128 (),
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvtepi16_epi64 (__m256i __W, __mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovsxwq256_mask ((__v8hi) __A,
                (__v4di) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepi16_epi64 (__mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovsxwq256_mask ((__v8hi) __A,
                (__v4di)
                _mm256_setzero_si256 (),
                (__mmask8) __U);
}


static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtepu8_epi32 (__m128i __W, __mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovzxbd128_mask ((__v16qi) __A,
                (__v4si) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtepu8_epi32 (__mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovzxbd128_mask ((__v16qi) __A,
                (__v4si)
                _mm_setzero_si128 (),
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvtepu8_epi32 (__m256i __W, __mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovzxbd256_mask ((__v16qi) __A,
                (__v8si) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepu8_epi32 (__mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovzxbd256_mask ((__v16qi) __A,
                (__v8si)
                _mm256_setzero_si256 (),
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtepu8_epi64 (__m128i __W, __mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovzxbq128_mask ((__v16qi) __A,
                (__v2di) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtepu8_epi64 (__mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovzxbq128_mask ((__v16qi) __A,
                (__v2di)
                _mm_setzero_si128 (),
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvtepu8_epi64 (__m256i __W, __mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovzxbq256_mask ((__v16qi) __A,
                (__v4di) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepu8_epi64 (__mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovzxbq256_mask ((__v16qi) __A,
                (__v4di)
                _mm256_setzero_si256 (),
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtepu32_epi64 (__m128i __W, __mmask8 __U, __m128i __X)
{
  return (__m128i) __builtin_ia32_pmovzxdq128_mask ((__v4si) __X,
                (__v2di) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtepu32_epi64 (__mmask8 __U, __m128i __X)
{
  return (__m128i) __builtin_ia32_pmovzxdq128_mask ((__v4si) __X,
                (__v2di)
                _mm_setzero_si128 (),
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvtepu32_epi64 (__m256i __W, __mmask8 __U, __m128i __X)
{
  return (__m256i) __builtin_ia32_pmovzxdq256_mask ((__v4si) __X,
                (__v4di) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepu32_epi64 (__mmask8 __U, __m128i __X)
{
  return (__m256i) __builtin_ia32_pmovzxdq256_mask ((__v4si) __X,
                (__v4di)
                _mm256_setzero_si256 (),
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtepu16_epi32 (__m128i __W, __mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovzxwd128_mask ((__v8hi) __A,
                (__v4si) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtepu16_epi32 (__mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovzxwd128_mask ((__v8hi) __A,
                (__v4si)
                _mm_setzero_si128 (),
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvtepu16_epi32 (__m256i __W, __mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovzxwd256_mask ((__v8hi) __A,
                (__v8si) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepu16_epi32 (__mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovzxwd256_mask ((__v8hi) __A,
                (__v8si)
                _mm256_setzero_si256 (),
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_cvtepu16_epi64 (__m128i __W, __mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovzxwq128_mask ((__v8hi) __A,
                (__v2di) __W,
                (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_cvtepu16_epi64 (__mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_pmovzxwq128_mask ((__v8hi) __A,
                (__v2di)
                _mm_setzero_si128 (),
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_cvtepu16_epi64 (__m256i __W, __mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovzxwq256_mask ((__v8hi) __A,
                (__v4di) __W,
                (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_cvtepu16_epi64 (__mmask8 __U, __m128i __A)
{
  return (__m256i) __builtin_ia32_pmovzxwq256_mask ((__v8hi) __A,
                (__v4di)
                _mm256_setzero_si256 (),
                (__mmask8) __U);
}


#define _mm_rol_epi32(a, b) __extension__ ({\
   (__m128i)__builtin_ia32_prold128_mask((__v4si) (a), (b),\
                                         (__v4si)\
                                         _mm_setzero_si128 (),\
                                         (__mmask8) -1); })

#define _mm_mask_rol_epi32(w, u, a, b) __extension__ ({\
   (__m128i)__builtin_ia32_prold128_mask((__v4si) (a), (b),\
                                         (__v4si) (w),\
                                         (__mmask8) (u)); })

#define _mm_maskz_rol_epi32(u, a, b) __extension__ ({\
   (__m128i)__builtin_ia32_prold128_mask((__v4si) (a), (b),\
                                         (__v4si)\
                                         _mm_setzero_si128 (),\
                                         (__mmask8) (u)); })

#define _mm256_rol_epi32(a, b) __extension__ ({\
   (__m256i)__builtin_ia32_prold256_mask((__v8si) (a), (b),\
                                         (__v8si)\
                                         _mm256_setzero_si256 (),\
                                         (__mmask8) -1); })

#define _mm256_mask_rol_epi32(w, u, a, b) __extension__ ({\
   (__m256i)__builtin_ia32_prold256_mask((__v8si) (a), (b),\
                                         (__v8si) (w),\
                                         (__mmask8) (u)); })

#define _mm256_maskz_rol_epi32(u, a, b) __extension__ ({\
   (__m256i)__builtin_ia32_prold256_mask((__v8si) (a), (b),\
                                         (__v8si)\
                                         _mm256_setzero_si256 (),\
                                         (__mmask8) (u)); })

#define _mm_rol_epi64(a, b) __extension__ ({\
   (__m128i)__builtin_ia32_prolq128_mask((__v2di) (a), (b),\
                                         (__v2di)\
                                         _mm_setzero_di (),\
                                         (__mmask8) -1); })

#define _mm_mask_rol_epi64(w, u, a, b) __extension__ ({\
   (__m128i)__builtin_ia32_prolq128_mask((__v2di) (a), (b),\
                                         (__v2di) (w),\
                                         (__mmask8) (u)); })

#define _mm_maskz_rol_epi64(u, a, b) __extension__ ({\
   (__m128i)__builtin_ia32_prolq128_mask((__v2di) (a), (b),\
                                         (__v2di)\
                                         _mm_setzero_di(),\
                                         (__mmask8) (u)); })

#define _mm256_rol_epi64(a, b) __extension__ ({\
   (__m256i)__builtin_ia32_prolq256_mask((__v4di) (a), (b),\
                                         (__v4di)\
                                         _mm256_setzero_si256 (),\
                                         (__mmask8) -1); })

#define _mm256_mask_rol_epi64(w, u, a, b) __extension__ ({\
   (__m256i)__builtin_ia32_prolq256_mask((__v4di) (a), (b),\
                                         (__v4di) (w),\
                                         (__mmask8) (u)); })

#define _mm256_maskz_rol_epi64(u, a, b) __extension__ ({\
   (__m256i)__builtin_ia32_prolq256_mask((__v4di) (a), (b),\
                                         (__v4di)\
                                         _mm256_setzero_si256 (),\
                                         (__mmask8) (u)); })

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_rolv_epi32 (__m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_prolvd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si)
              _mm_setzero_si128 (),
              (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_rolv_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
         __m128i __B)
{
  return (__m128i) __builtin_ia32_prolvd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si) __W,
              (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_rolv_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_prolvd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si)
              _mm_setzero_si128 (),
              (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_rolv_epi32 (__m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_prolvd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si)
              _mm256_setzero_si256 (),
              (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_rolv_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
      __m256i __B)
{
  return (__m256i) __builtin_ia32_prolvd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si) __W,
              (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_rolv_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_prolvd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si)
              _mm256_setzero_si256 (),
              (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_rolv_epi64 (__m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_prolvq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_di (),
              (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_rolv_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
         __m128i __B)
{
  return (__m128i) __builtin_ia32_prolvq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di) __W,
              (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_rolv_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_prolvq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_di (),
              (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_rolv_epi64 (__m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_prolvq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_si256 (),
              (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_rolv_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
      __m256i __B)
{
  return (__m256i) __builtin_ia32_prolvq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di) __W,
              (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_rolv_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_prolvq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_si256 (),
              (__mmask8) __U);
}

#define _mm_ror_epi32( __A, __B) __extension__ ({ \
__builtin_ia32_prord128_mask ((__v4si)( __A),( __B),\
             (__v4si)\
             _mm_setzero_si128 (),\
             (__mmask8) -1);\
})

#define _mm_mask_ror_epi32( __W, __U, __A ,__B) __extension__ ({ \
__builtin_ia32_prord128_mask ((__v4si) __A, __B,\
             (__v4si)( __W),\
             (__mmask8)( __U));\
})

#define _mm_maskz_ror_epi32( __U, __A, __B) __extension__ ({ \
__builtin_ia32_prord128_mask ((__v4si)( __A),( __B),\
             (__v4si)\
             _mm_setzero_si128 (),\
             (__mmask8)( __U));\
})

#define _mm256_ror_epi32( __A, __B) __extension__ ({ \
__builtin_ia32_prord256_mask ((__v8si)( __A),( __B),\
             (__v8si)\
             _mm256_setzero_si256 (),\
             (__mmask8) -1);\
})

#define _mm256_mask_ror_epi32( __W, __U, __A ,__B) __extension__ ({ \
__builtin_ia32_prord256_mask ((__v8si) __A, __B,\
             (__v8si)( __W),\
             (__mmask8)( __U));\
})

#define _mm256_maskz_ror_epi32( __U, __A, __B) __extension__ ({ \
__builtin_ia32_prord256_mask ((__v8si)( __A),( __B),\
             (__v8si)\
             _mm256_setzero_si256 (),\
             (__mmask8)( __U));\
})

#define _mm_ror_epi64( __A, __B) __extension__ ({ \
__builtin_ia32_prorq128_mask ((__v2di)( __A),( __B),\
             (__v2di)\
             _mm_setzero_di (),\
             (__mmask8) -1);\
})

#define _mm_mask_ror_epi64( __W, __U, __A ,__B) __extension__ ({ \
__builtin_ia32_prorq128_mask ((__v2di) __A, __B,\
             (__v2di)( __W),\
             (__mmask8)( __U));\
})

#define _mm_maskz_ror_epi64( __U, __A, __B) __extension__ ({ \
__builtin_ia32_prorq128_mask ((__v2di)( __A),( __B),\
             (__v2di)\
             _mm_setzero_di (),\
             (__mmask8)( __U));\
})

#define _mm256_ror_epi64( __A, __B) __extension__ ({ \
__builtin_ia32_prorq256_mask ((__v4di)( __A),( __B),\
             (__v4di)\
             _mm256_setzero_si256 (),\
             (__mmask8) -1);\
})

#define _mm256_mask_ror_epi64( __W, __U, __A ,__B) __extension__ ({ \
__builtin_ia32_prorq256_mask ((__v4di) __A, __B,\
             (__v4di)( __W),\
             (__mmask8)( __U));\
})

#define _mm256_maskz_ror_epi64( __U, __A, __B) __extension__ ({ \
__builtin_ia32_prorq256_mask ((__v4di)( __A),( __B),\
             (__v4di)\
             _mm256_setzero_si256 (),\
             (__mmask8)( __U));\
})

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_sll_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_pslld128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_sll_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pslld128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_sll_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m128i __B)
{
  return (__m256i) __builtin_ia32_pslld256_mask ((__v8si) __A,
             (__v4si) __B,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_sll_epi32 (__mmask8 __U, __m256i __A, __m128i __B)
{
  return (__m256i) __builtin_ia32_pslld256_mask ((__v8si) __A,
             (__v4si) __B,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

#define _mm_mask_slli_epi32( __W, __U, __A, __B) __extension__ ({ \
__builtin_ia32_pslldi128_mask ((__v4si)( __A),( __B),\
              (__v4si)( __W),\
              (__mmask8)( __U));\
})

#define _mm_maskz_slli_epi32( __U, __A, __B) __extension__ ({ \
__builtin_ia32_pslldi128_mask ((__v4si)( __A),( __B),\
              (__v4si)\
              _mm_setzero_si128 (),\
              (__mmask8)( __U));\
})

#define _mm256_mask_slli_epi32( __W, __U, __A ,__B) __extension__ ({ \
__builtin_ia32_pslldi256_mask ((__v8si) (__A), (__B),\
              (__v8si)( __W),\
              (__mmask8)( __U));\
})

#define _mm256_maskz_slli_epi32( __U, __A, __B) __extension__ ({ \
__builtin_ia32_pslldi256_mask ((__v8si)( __A),( __B),\
              (__v8si)\
              _mm256_setzero_si256 (),\
              (__mmask8)( __U));\
})

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_sll_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_psllq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_sll_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_psllq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di)
             _mm_setzero_di (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_sll_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m128i __B)
{
  return (__m256i) __builtin_ia32_psllq256_mask ((__v4di) __A,
             (__v2di) __B,
             (__v4di) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_sll_epi64 (__mmask8 __U, __m256i __A, __m128i __B)
{
  return (__m256i) __builtin_ia32_psllq256_mask ((__v4di) __A,
             (__v2di) __B,
             (__v4di)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

#define _mm_mask_slli_epi64( __W, __U, __A, __B) __extension__ ({ \
__builtin_ia32_psllqi128_mask ((__v2di)( __A),( __B),\
              (__v2di)( __W),\
              (__mmask8)( __U));\
})

#define _mm_maskz_slli_epi64( __U, __A, __B) __extension__ ({ \
__builtin_ia32_psllqi128_mask ((__v2di)( __A),( __B),\
              (__v2di)\
              _mm_setzero_di (),\
              (__mmask8)( __U));\
})

#define _mm256_mask_slli_epi64( __W, __U, __A ,__B) __extension__ ({ \
__builtin_ia32_psllqi256_mask ((__v4di) (__A), (__B),\
              (__v4di)( __W),\
              (__mmask8)( __U));\
})

#define _mm256_maskz_slli_epi64( __U, __A, __B) __extension__ ({ \
__builtin_ia32_psllqi256_mask ((__v4di)( __A),( __B),\
              (__v4di)\
              _mm256_setzero_si256 (),\
              (__mmask8)( __U));\
})


static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_rorv_epi32 (__m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_prorvd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si)
              _mm_setzero_si128 (),
              (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_rorv_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
         __m128i __B)
{
  return (__m128i) __builtin_ia32_prorvd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si) __W,
              (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_rorv_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_prorvd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si)
              _mm_setzero_si128 (),
              (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_rorv_epi32 (__m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_prorvd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si)
              _mm256_setzero_si256 (),
              (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_rorv_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
      __m256i __B)
{
  return (__m256i) __builtin_ia32_prorvd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si) __W,
              (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_rorv_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_prorvd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si)
              _mm256_setzero_si256 (),
              (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_rorv_epi64 (__m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_prorvq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_di (),
              (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_rorv_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
         __m128i __B)
{
  return (__m128i) __builtin_ia32_prorvq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di) __W,
              (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_rorv_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_prorvq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_di (),
              (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_rorv_epi64 (__m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_prorvq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_si256 (),
              (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_rorv_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
      __m256i __B)
{
  return (__m256i) __builtin_ia32_prorvq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di) __W,
              (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_rorv_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_prorvq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_si256 (),
              (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_sllv_epi64 (__m128i __W, __mmask8 __U, __m128i __X,
         __m128i __Y)
{
  return (__m128i) __builtin_ia32_psllv2di_mask ((__v2di) __X,
             (__v2di) __Y,
             (__v2di) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_sllv_epi64 (__mmask8 __U, __m128i __X, __m128i __Y)
{
  return (__m128i) __builtin_ia32_psllv2di_mask ((__v2di) __X,
             (__v2di) __Y,
             (__v2di)
             _mm_setzero_di (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_sllv_epi64 (__m256i __W, __mmask8 __U, __m256i __X,
      __m256i __Y)
{
  return (__m256i) __builtin_ia32_psllv4di_mask ((__v4di) __X,
             (__v4di) __Y,
             (__v4di) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_sllv_epi64 (__mmask8 __U, __m256i __X, __m256i __Y)
{
  return (__m256i) __builtin_ia32_psllv4di_mask ((__v4di) __X,
             (__v4di) __Y,
             (__v4di)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_sllv_epi32 (__m128i __W, __mmask8 __U, __m128i __X,
         __m128i __Y)
{
  return (__m128i) __builtin_ia32_psllv4si_mask ((__v4si) __X,
             (__v4si) __Y,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_sllv_epi32 (__mmask8 __U, __m128i __X, __m128i __Y)
{
  return (__m128i) __builtin_ia32_psllv4si_mask ((__v4si) __X,
             (__v4si) __Y,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_sllv_epi32 (__m256i __W, __mmask8 __U, __m256i __X,
      __m256i __Y)
{
  return (__m256i) __builtin_ia32_psllv8si_mask ((__v8si) __X,
             (__v8si) __Y,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_sllv_epi32 (__mmask8 __U, __m256i __X, __m256i __Y)
{
  return (__m256i) __builtin_ia32_psllv8si_mask ((__v8si) __X,
             (__v8si) __Y,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

#define _mm256_maskz_sllv_epi32( __U, __X, __Y) __extension__ ({ \
__builtin_ia32_psllv8si_mask ((__v8si)( __X),\
             (__v8si)( __Y),\
             (__v8si)\
             _mm256_setzero_si256 (),\
             (__mmask8)( __U));\
})



static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_srlv_epi64 (__m128i __W, __mmask8 __U, __m128i __X,
         __m128i __Y)
{
  return (__m128i) __builtin_ia32_psrlv2di_mask ((__v2di) __X,
             (__v2di) __Y,
             (__v2di) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_srlv_epi64 (__mmask8 __U, __m128i __X, __m128i __Y)
{
  return (__m128i) __builtin_ia32_psrlv2di_mask ((__v2di) __X,
             (__v2di) __Y,
             (__v2di)
             _mm_setzero_di (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_srlv_epi64 (__m256i __W, __mmask8 __U, __m256i __X,
      __m256i __Y)
{
  return (__m256i) __builtin_ia32_psrlv4di_mask ((__v4di) __X,
             (__v4di) __Y,
             (__v4di) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_srlv_epi64 (__mmask8 __U, __m256i __X, __m256i __Y)
{
  return (__m256i) __builtin_ia32_psrlv4di_mask ((__v4di) __X,
             (__v4di) __Y,
             (__v4di)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_srlv_epi32 (__m128i __W, __mmask8 __U, __m128i __X,
         __m128i __Y)
{
  return (__m128i) __builtin_ia32_psrlv4si_mask ((__v4si) __X,
             (__v4si) __Y,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_srlv_epi32 (__mmask8 __U, __m128i __X, __m128i __Y)
{
  return (__m128i) __builtin_ia32_psrlv4si_mask ((__v4si) __X,
             (__v4si) __Y,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_srlv_epi32 (__m256i __W, __mmask8 __U, __m256i __X,
      __m256i __Y)
{
  return (__m256i) __builtin_ia32_psrlv8si_mask ((__v8si) __X,
             (__v8si) __Y,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_srlv_epi32 (__mmask8 __U, __m256i __X, __m256i __Y)
{
  return (__m256i) __builtin_ia32_psrlv8si_mask ((__v8si) __X,
             (__v8si) __Y,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}



static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_srl_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_psrld128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_srl_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_psrld128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_srl_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m128i __B)
{
  return (__m256i) __builtin_ia32_psrld256_mask ((__v8si) __A,
             (__v4si) __B,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_srl_epi32 (__mmask8 __U, __m256i __A, __m128i __B)
{
  return (__m256i) __builtin_ia32_psrld256_mask ((__v8si) __A,
             (__v4si) __B,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

#define _mm_mask_srli_epi32( __W, __U, __A, __imm) __extension__ ({ \
__builtin_ia32_psrldi128_mask ((__v4si)( __A),( __imm),\
              (__v4si)( __W),\
              (__mmask8)( __U));\
})

#define _mm_maskz_srli_epi32( __U, __A, __imm) __extension__ ({ \
__builtin_ia32_psrldi128_mask ((__v4si)( __A),( __imm),\
              (__v4si)\
              _mm_setzero_si128 (),\
              (__mmask8)( __U));\
})

#define _mm256_mask_srli_epi32( __W, __U, __A, __imm) __extension__ ({ \
__builtin_ia32_psrldi256_mask ((__v8si)( __A),( __imm),\
              (__v8si)( __W),\
              (__mmask8)( __U));\
})

#define _mm256_maskz_srli_epi32( __U, __A, __imm) __extension__ ({ \
__builtin_ia32_psrldi256_mask ((__v8si)( __A),( __imm),\
              (__v8si)\
              _mm256_setzero_si256 (),\
              (__mmask8)( __U));\
})

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_srl_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_psrlq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_srl_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_psrlq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di)
             _mm_setzero_di (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_srl_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m128i __B)
{
  return (__m256i) __builtin_ia32_psrlq256_mask ((__v4di) __A,
             (__v2di) __B,
             (__v4di) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_srl_epi64 (__mmask8 __U, __m256i __A, __m128i __B)
{
  return (__m256i) __builtin_ia32_psrlq256_mask ((__v4di) __A,
             (__v2di) __B,
             (__v4di)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

#define _mm_mask_srli_epi64( __W, __U, __A, __imm) __extension__ ({ \
__builtin_ia32_psrlqi128_mask ((__v2di)( __A),( __imm),\
              (__v2di)( __W),\
              (__mmask8)( __U));\
})

#define _mm_maskz_srli_epi64( __U, __A, __imm) __extension__ ({ \
__builtin_ia32_psrlqi128_mask ((__v2di)( __A),( __imm),\
              (__v2di)\
              _mm_setzero_si128 (),\
              (__mmask8)( __U));\
})

#define _mm256_mask_srli_epi64( __W, __U, __A, __imm) __extension__ ({ \
__builtin_ia32_psrlqi256_mask ((__v4di)( __A),( __imm),\
              (__v4di)( __W),\
              (__mmask8)( __U));\
})

#define _mm256_maskz_srli_epi64( __U, __A, __imm) __extension__ ({ \
__builtin_ia32_psrlqi256_mask ((__v4di)( __A),( __imm),\
              (__v4di)\
              _mm256_setzero_si256 (),\
              (__mmask8)( __U));\
})

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_srav_epi32 (__m128i __W, __mmask8 __U, __m128i __X,
         __m128i __Y)
{
  return (__m128i) __builtin_ia32_psrav4si_mask ((__v4si) __X,
             (__v4si) __Y,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_srav_epi32 (__mmask8 __U, __m128i __X, __m128i __Y)
{
  return (__m128i) __builtin_ia32_psrav4si_mask ((__v4si) __X,
             (__v4si) __Y,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_srav_epi32 (__m256i __W, __mmask8 __U, __m256i __X,
      __m256i __Y)
{
  return (__m256i) __builtin_ia32_psrav8si_mask ((__v8si) __X,
             (__v8si) __Y,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_srav_epi32 (__mmask8 __U, __m256i __X, __m256i __Y)
{
  return (__m256i) __builtin_ia32_psrav8si_mask ((__v8si) __X,
             (__v8si) __Y,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srav_epi64 (__m128i __X, __m128i __Y)
{
  return (__m128i) __builtin_ia32_psravq128_mask ((__v2di) __X,
              (__v2di) __Y,
              (__v2di)
              _mm_setzero_di (),
              (__mmask8) -1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_srav_epi64 (__m128i __W, __mmask8 __U, __m128i __X,
         __m128i __Y)
{
  return (__m128i) __builtin_ia32_psravq128_mask ((__v2di) __X,
              (__v2di) __Y,
              (__v2di) __W,
              (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_srav_epi64 (__mmask8 __U, __m128i __X, __m128i __Y)
{
  return (__m128i) __builtin_ia32_psravq128_mask ((__v2di) __X,
              (__v2di) __Y,
              (__v2di)
              _mm_setzero_di (),
              (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_srav_epi64 (__m256i __X, __m256i __Y)
{
  return (__m256i) __builtin_ia32_psravq256_mask ((__v4di) __X,
              (__v4di) __Y,
              (__v4di)
              _mm256_setzero_si256 (),
              (__mmask8) -1);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_srav_epi64 (__m256i __W, __mmask8 __U, __m256i __X,
      __m256i __Y)
{
  return (__m256i) __builtin_ia32_psravq256_mask ((__v4di) __X,
              (__v4di) __Y,
              (__v4di) __W,
              (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_srav_epi64 (__mmask8 __U, __m256i __X, __m256i __Y)
{
  return (__m256i) __builtin_ia32_psravq256_mask ((__v4di) __X,
              (__v4di) __Y,
              (__v4di)
              _mm256_setzero_si256 (),
              (__mmask8) __U);
}



static __inline__ void __DEFAULT_FN_ATTRS
_mm_mask_store_epi32 (void *__P, __mmask8 __U, __m128i __A)
{
  __builtin_ia32_movdqa32store128_mask ((__v4si *) __P,
          (__v4si) __A,
          (__mmask8) __U);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm256_mask_store_epi32 (void *__P, __mmask8 __U, __m256i __A)
{
  __builtin_ia32_movdqa32store256_mask ((__v8si *) __P,
          (__v8si) __A,
          (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_mov_epi64 (__m128i __W, __mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_movdqa64_128_mask ((__v2di) __A,
                 (__v2di) __W,
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_mov_epi64 (__mmask8 __U, __m128i __A)
{
  return (__m128i) __builtin_ia32_movdqa64_128_mask ((__v2di) __A,
                 (__v2di)
                 _mm_setzero_di (),
                 (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_mov_epi64 (__m256i __W, __mmask8 __U, __m256i __A)
{
  return (__m256i) __builtin_ia32_movdqa64_256_mask ((__v4di) __A,
                 (__v4di) __W,
                 (__mmask8) __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_mov_epi64 (__mmask8 __U, __m256i __A)
{
  return (__m256i) __builtin_ia32_movdqa64_256_mask ((__v4di) __A,
                 (__v4di)
                 _mm256_setzero_si256 (),
                 (__mmask8) __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_load_epi64 (__m128i __W, __mmask8 __U, void const *__P)
{
  return (__m128i) __builtin_ia32_movdqa64load128_mask ((__v2di *) __P,
              (__v2di) __W,
              (__mmask8)
              __U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_load_epi64 (__mmask8 __U, void const *__P)
{
  return (__m128i) __builtin_ia32_movdqa64load128_mask ((__v2di *) __P,
              (__v2di)
              _mm_setzero_di (),
              (__mmask8)
              __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_load_epi64 (__m256i __W, __mmask8 __U, void const *__P)
{
  return (__m256i) __builtin_ia32_movdqa64load256_mask ((__v4di *) __P,
              (__v4di) __W,
              (__mmask8)
              __U);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_load_epi64 (__mmask8 __U, void const *__P)
{
  return (__m256i) __builtin_ia32_movdqa64load256_mask ((__v4di *) __P,
              (__v4di)
              _mm256_setzero_si256 (),
              (__mmask8)
              __U);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_mask_store_epi64 (void *__P, __mmask8 __U, __m128i __A)
{
  __builtin_ia32_movdqa64store128_mask ((__v2di *) __P,
          (__v2di) __A,
          (__mmask8) __U);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm256_mask_store_epi64 (void *__P, __mmask8 __U, __m256i __A)
{
  __builtin_ia32_movdqa64store256_mask ((__v4di *) __P,
          (__v4di) __A,
          (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mask_movedup_pd (__m128d __W, __mmask8 __U, __m128d __A)
{
  return (__m128d) __builtin_ia32_movddup128_mask ((__v2df) __A,
               (__v2df) __W,
               (__mmask8) __U);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_maskz_movedup_pd (__mmask8 __U, __m128d __A)
{
  return (__m128d) __builtin_ia32_movddup128_mask ((__v2df) __A,
               (__v2df)
               _mm_setzero_pd (),
               (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_mask_movedup_pd (__m256d __W, __mmask8 __U, __m256d __A)
{
  return (__m256d) __builtin_ia32_movddup256_mask ((__v4df) __A,
               (__v4df) __W,
               (__mmask8) __U);
}

static __inline__ __m256d __DEFAULT_FN_ATTRS
_mm256_maskz_movedup_pd (__mmask8 __U, __m256d __A)
{
  return (__m256d) __builtin_ia32_movddup256_mask ((__v4df) __A,
               (__v4df)
              _mm256_setzero_pd (),
               (__mmask8) __U);
}


#define _mm_mask_set1_epi32( __O, __M, __A) __extension__ ({ \
__builtin_ia32_pbroadcastd128_gpr_mask (__A, (__v4si)( __O),\
                ( __M));\
})

#define _mm_maskz_set1_epi32( __M, __A) __extension__ ({ \
__builtin_ia32_pbroadcastd128_gpr_mask (__A,\
                 (__v4si)\
                 _mm_setzero_si128 (),\
                ( __M));\
})

#define _mm256_mask_set1_epi32( __O, __M, __A) __extension__ ({ \
__builtin_ia32_pbroadcastd256_gpr_mask (__A, (__v8si)( __O),\
                ( __M));\
})

#define _mm256_maskz_set1_epi32( __M, __A) __extension__ ({ \
__builtin_ia32_pbroadcastd256_gpr_mask (__A,\
                 (__v8si)\
                 _mm256_setzero_si256 (),\
                ( __M));\
})

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mask_set1_epi64 (__m128i __O, __mmask8 __M, long long __A)
{
  return (__m128i) __builtin_ia32_pbroadcastq128_gpr_mask (__A, (__v2di) __O,
                 __M);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_maskz_set1_epi64 (__mmask8 __M, long long __A)
{
  return (__m128i) __builtin_ia32_pbroadcastq128_gpr_mask (__A,
                 (__v2di)
                 _mm_setzero_si128 (),
                 __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_mask_set1_epi64 (__m256i __O, __mmask8 __M, long long __A)
{
  return (__m256i) __builtin_ia32_pbroadcastq256_gpr_mask (__A, (__v4di) __O,
                 __M);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_maskz_set1_epi64 (__mmask8 __M, long long __A)
{
  return (__m256i) __builtin_ia32_pbroadcastq256_gpr_mask (__A,
                 (__v4di)
                 _mm256_setzero_si256 (),
                 __M);
}
#undef __DEFAULT_FN_ATTRS
#undef __DEFAULT_FN_ATTRS_BOTH

#endif /* __AVX512VLINTRIN_H */
