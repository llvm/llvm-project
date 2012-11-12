/*===---- avxintrin.h - AVX intrinsics -------------------------------------===
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
#error "Never use <avxintrin.h> directly; include <immintrin.h> instead."
#endif

typedef double __v4df __attribute__ ((__vector_size__ (32)));
typedef float __v8sf __attribute__ ((__vector_size__ (32)));
typedef long long __v4di __attribute__ ((__vector_size__ (32)));
typedef int __v8si __attribute__ ((__vector_size__ (32)));
typedef short __v16hi __attribute__ ((__vector_size__ (32)));
typedef char __v32qi __attribute__ ((__vector_size__ (32)));

typedef float __m256 __attribute__ ((__vector_size__ (32)));
typedef double __m256d __attribute__((__vector_size__(32)));
typedef long long __m256i __attribute__((__vector_size__(32)));

/* Arithmetic */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_add_pd(__m256d a, __m256d b)
{
  return a+b;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_add_ps(__m256 a, __m256 b)
{
  return a+b;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_sub_pd(__m256d a, __m256d b)
{
  return a-b;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_sub_ps(__m256 a, __m256 b)
{
  return a-b;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_addsub_pd(__m256d a, __m256d b)
{
  return (__m256d)__builtin_ia32_addsubpd256((__v4df)a, (__v4df)b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_addsub_ps(__m256 a, __m256 b)
{
  return (__m256)__builtin_ia32_addsubps256((__v8sf)a, (__v8sf)b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_div_pd(__m256d a, __m256d b)
{
  return a / b;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_div_ps(__m256 a, __m256 b)
{
  return a / b;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_max_pd(__m256d a, __m256d b)
{
  return (__m256d)__builtin_ia32_maxpd256((__v4df)a, (__v4df)b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_max_ps(__m256 a, __m256 b)
{
  return (__m256)__builtin_ia32_maxps256((__v8sf)a, (__v8sf)b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_min_pd(__m256d a, __m256d b)
{
  return (__m256d)__builtin_ia32_minpd256((__v4df)a, (__v4df)b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_min_ps(__m256 a, __m256 b)
{
  return (__m256)__builtin_ia32_minps256((__v8sf)a, (__v8sf)b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_mul_pd(__m256d a, __m256d b)
{
  return a * b;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_mul_ps(__m256 a, __m256 b)
{
  return a * b;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_sqrt_pd(__m256d a)
{
  return (__m256d)__builtin_ia32_sqrtpd256((__v4df)a);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_sqrt_ps(__m256 a)
{
  return (__m256)__builtin_ia32_sqrtps256((__v8sf)a);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_rsqrt_ps(__m256 a)
{
  return (__m256)__builtin_ia32_rsqrtps256((__v8sf)a);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_rcp_ps(__m256 a)
{
  return (__m256)__builtin_ia32_rcpps256((__v8sf)a);
}

#define _mm256_round_pd(V, M) __extension__ ({ \
    __m256d __V = (V); \
    (__m256d)__builtin_ia32_roundpd256((__v4df)__V, (M)); })

#define _mm256_round_ps(V, M) __extension__ ({ \
  __m256 __V = (V); \
  (__m256)__builtin_ia32_roundps256((__v8sf)__V, (M)); })

#define _mm256_ceil_pd(V)  _mm256_round_pd((V), _MM_FROUND_CEIL)
#define _mm256_floor_pd(V) _mm256_round_pd((V), _MM_FROUND_FLOOR)
#define _mm256_ceil_ps(V)  _mm256_round_ps((V), _MM_FROUND_CEIL)
#define _mm256_floor_ps(V) _mm256_round_ps((V), _MM_FROUND_FLOOR)

/* Logical */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_and_pd(__m256d a, __m256d b)
{
  return (__m256d)((__v4di)a & (__v4di)b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_and_ps(__m256 a, __m256 b)
{
  return (__m256)((__v8si)a & (__v8si)b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_andnot_pd(__m256d a, __m256d b)
{
  return (__m256d)(~(__v4di)a & (__v4di)b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_andnot_ps(__m256 a, __m256 b)
{
  return (__m256)(~(__v8si)a & (__v8si)b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_or_pd(__m256d a, __m256d b)
{
  return (__m256d)((__v4di)a | (__v4di)b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_or_ps(__m256 a, __m256 b)
{
  return (__m256)((__v8si)a | (__v8si)b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_xor_pd(__m256d a, __m256d b)
{
  return (__m256d)((__v4di)a ^ (__v4di)b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_xor_ps(__m256 a, __m256 b)
{
  return (__m256)((__v8si)a ^ (__v8si)b);
}

/* Horizontal arithmetic */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_hadd_pd(__m256d a, __m256d b)
{
  return (__m256d)__builtin_ia32_haddpd256((__v4df)a, (__v4df)b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_hadd_ps(__m256 a, __m256 b)
{
  return (__m256)__builtin_ia32_haddps256((__v8sf)a, (__v8sf)b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_hsub_pd(__m256d a, __m256d b)
{
  return (__m256d)__builtin_ia32_hsubpd256((__v4df)a, (__v4df)b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_hsub_ps(__m256 a, __m256 b)
{
  return (__m256)__builtin_ia32_hsubps256((__v8sf)a, (__v8sf)b);
}

/* Vector permutations */
static __inline __m128d __attribute__((__always_inline__, __nodebug__))
_mm_permutevar_pd(__m128d a, __m128i c)
{
  return (__m128d)__builtin_ia32_vpermilvarpd((__v2df)a, (__v2di)c);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_permutevar_pd(__m256d a, __m256i c)
{
  return (__m256d)__builtin_ia32_vpermilvarpd256((__v4df)a, (__v4di)c);
}

static __inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_permutevar_ps(__m128 a, __m128i c)
{
  return (__m128)__builtin_ia32_vpermilvarps((__v4sf)a, (__v4si)c);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_permutevar_ps(__m256 a, __m256i c)
{
  return (__m256)__builtin_ia32_vpermilvarps256((__v8sf)a,
						  (__v8si)c);
}

#define _mm_permute_pd(A, C) __extension__ ({ \
  __m128d __A = (A); \
  (__m128d)__builtin_shufflevector((__v2df)__A, (__v2df) _mm_setzero_pd(), \
                                   (C) & 0x1, ((C) & 0x2) >> 1); })

#define _mm256_permute_pd(A, C) __extension__ ({ \
  __m256d __A = (A); \
  (__m256d)__builtin_shufflevector((__v4df)__A, (__v4df) _mm256_setzero_pd(), \
                                   (C) & 0x1, ((C) & 0x2) >> 1, \
                                   2 + (((C) & 0x4) >> 2), \
                                   2 + (((C) & 0x8) >> 3)); })

#define _mm_permute_ps(A, C) __extension__ ({ \
  __m128 __A = (A); \
  (__m128)__builtin_shufflevector((__v4sf)__A, (__v4sf) _mm_setzero_ps(), \
                                   (C) & 0x3, ((C) & 0xc) >> 2, \
                                   ((C) & 0x30) >> 4, ((C) & 0xc0) >> 6); })

#define _mm256_permute_ps(A, C) __extension__ ({ \
  __m256 __A = (A); \
  (__m256)__builtin_shufflevector((__v8sf)__A, (__v8sf) _mm256_setzero_ps(), \
                                  (C) & 0x3, ((C) & 0xc) >> 2, \
                                  ((C) & 0x30) >> 4, ((C) & 0xc0) >> 6, \
                                  4 + (((C) & 0x03) >> 0), \
                                  4 + (((C) & 0x0c) >> 2), \
                                  4 + (((C) & 0x30) >> 4), \
                                  4 + (((C) & 0xc0) >> 6)); })

#define _mm256_permute2f128_pd(V1, V2, M) __extension__ ({ \
  __m256d __V1 = (V1); \
  __m256d __V2 = (V2); \
  (__m256d)__builtin_ia32_vperm2f128_pd256((__v4df)__V1, (__v4df)__V2, (M)); })

#define _mm256_permute2f128_ps(V1, V2, M) __extension__ ({ \
  __m256 __V1 = (V1); \
  __m256 __V2 = (V2); \
  (__m256)__builtin_ia32_vperm2f128_ps256((__v8sf)__V1, (__v8sf)__V2, (M)); })

#define _mm256_permute2f128_si256(V1, V2, M) __extension__ ({ \
  __m256i __V1 = (V1); \
  __m256i __V2 = (V2); \
  (__m256i)__builtin_ia32_vperm2f128_si256((__v8si)__V1, (__v8si)__V2, (M)); })

/* Vector Blend */
#define _mm256_blend_pd(V1, V2, M) __extension__ ({ \
  __m256d __V1 = (V1); \
  __m256d __V2 = (V2); \
  (__m256d)__builtin_ia32_blendpd256((__v4df)__V1, (__v4df)__V2, (M)); })

#define _mm256_blend_ps(V1, V2, M) __extension__ ({ \
  __m256 __V1 = (V1); \
  __m256 __V2 = (V2); \
  (__m256)__builtin_ia32_blendps256((__v8sf)__V1, (__v8sf)__V2, (M)); })

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_blendv_pd(__m256d a, __m256d b, __m256d c)
{
  return (__m256d)__builtin_ia32_blendvpd256((__v4df)a, (__v4df)b, (__v4df)c);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_blendv_ps(__m256 a, __m256 b, __m256 c)
{
  return (__m256)__builtin_ia32_blendvps256((__v8sf)a, (__v8sf)b, (__v8sf)c);
}

/* Vector Dot Product */
#define _mm256_dp_ps(V1, V2, M) __extension__ ({ \
  __m256 __V1 = (V1); \
  __m256 __V2 = (V2); \
  (__m256)__builtin_ia32_dpps256((__v8sf)__V1, (__v8sf)__V2, (M)); })

/* Vector shuffle */
#define _mm256_shuffle_ps(a, b, mask) __extension__ ({ \
        __m256 __a = (a); \
        __m256 __b = (b); \
        (__m256)__builtin_shufflevector((__v8sf)__a, (__v8sf)__b, \
        (mask) & 0x3,                ((mask) & 0xc) >> 2, \
        (((mask) & 0x30) >> 4) + 8,  (((mask) & 0xc0) >> 6) + 8, \
        ((mask) & 0x3) + 4,          (((mask) & 0xc) >> 2) + 4, \
        (((mask) & 0x30) >> 4) + 12, (((mask) & 0xc0) >> 6) + 12); })

#define _mm256_shuffle_pd(a, b, mask) __extension__ ({ \
        __m256d __a = (a); \
        __m256d __b = (b); \
        (__m256d)__builtin_shufflevector((__v4df)__a, (__v4df)__b, \
        (mask) & 0x1, \
        (((mask) & 0x2) >> 1) + 4, \
        (((mask) & 0x4) >> 2) + 2, \
        (((mask) & 0x8) >> 3) + 6); })

/* Compare */
#define _CMP_EQ_OQ    0x00 /* Equal (ordered, non-signaling)  */
#define _CMP_LT_OS    0x01 /* Less-than (ordered, signaling)  */
#define _CMP_LE_OS    0x02 /* Less-than-or-equal (ordered, signaling)  */
#define _CMP_UNORD_Q  0x03 /* Unordered (non-signaling)  */
#define _CMP_NEQ_UQ   0x04 /* Not-equal (unordered, non-signaling)  */
#define _CMP_NLT_US   0x05 /* Not-less-than (unordered, signaling)  */
#define _CMP_NLE_US   0x06 /* Not-less-than-or-equal (unordered, signaling)  */
#define _CMP_ORD_Q    0x07 /* Ordered (nonsignaling)   */
#define _CMP_EQ_UQ    0x08 /* Equal (unordered, non-signaling)  */
#define _CMP_NGE_US   0x09 /* Not-greater-than-or-equal (unord, signaling)  */
#define _CMP_NGT_US   0x0a /* Not-greater-than (unordered, signaling)  */
#define _CMP_FALSE_OQ 0x0b /* False (ordered, non-signaling)  */
#define _CMP_NEQ_OQ   0x0c /* Not-equal (ordered, non-signaling)  */
#define _CMP_GE_OS    0x0d /* Greater-than-or-equal (ordered, signaling)  */
#define _CMP_GT_OS    0x0e /* Greater-than (ordered, signaling)  */
#define _CMP_TRUE_UQ  0x0f /* True (unordered, non-signaling)  */
#define _CMP_EQ_OS    0x10 /* Equal (ordered, signaling)  */
#define _CMP_LT_OQ    0x11 /* Less-than (ordered, non-signaling)  */
#define _CMP_LE_OQ    0x12 /* Less-than-or-equal (ordered, non-signaling)  */
#define _CMP_UNORD_S  0x13 /* Unordered (signaling)  */
#define _CMP_NEQ_US   0x14 /* Not-equal (unordered, signaling)  */
#define _CMP_NLT_UQ   0x15 /* Not-less-than (unordered, non-signaling)  */
#define _CMP_NLE_UQ   0x16 /* Not-less-than-or-equal (unord, non-signaling)  */
#define _CMP_ORD_S    0x17 /* Ordered (signaling)  */
#define _CMP_EQ_US    0x18 /* Equal (unordered, signaling)  */
#define _CMP_NGE_UQ   0x19 /* Not-greater-than-or-equal (unord, non-sign)  */
#define _CMP_NGT_UQ   0x1a /* Not-greater-than (unordered, non-signaling)  */
#define _CMP_FALSE_OS 0x1b /* False (ordered, signaling)  */
#define _CMP_NEQ_OS   0x1c /* Not-equal (ordered, signaling)  */
#define _CMP_GE_OQ    0x1d /* Greater-than-or-equal (ordered, non-signaling)  */
#define _CMP_GT_OQ    0x1e /* Greater-than (ordered, non-signaling)  */
#define _CMP_TRUE_US  0x1f /* True (unordered, signaling)  */

#define _mm_cmp_pd(a, b, c) __extension__ ({ \
  __m128d __a = (a); \
  __m128d __b = (b); \
  (__m128d)__builtin_ia32_cmppd((__v2df)__a, (__v2df)__b, (c)); })

#define _mm_cmp_ps(a, b, c) __extension__ ({ \
  __m128 __a = (a); \
  __m128 __b = (b); \
  (__m128)__builtin_ia32_cmpps((__v4sf)__a, (__v4sf)__b, (c)); })

#define _mm256_cmp_pd(a, b, c) __extension__ ({ \
  __m256d __a = (a); \
  __m256d __b = (b); \
  (__m256d)__builtin_ia32_cmppd256((__v4df)__a, (__v4df)__b, (c)); })

#define _mm256_cmp_ps(a, b, c) __extension__ ({ \
  __m256 __a = (a); \
  __m256 __b = (b); \
  (__m256)__builtin_ia32_cmpps256((__v8sf)__a, (__v8sf)__b, (c)); })

#define _mm_cmp_sd(a, b, c) __extension__ ({ \
  __m128d __a = (a); \
  __m128d __b = (b); \
  (__m128d)__builtin_ia32_cmpsd((__v2df)__a, (__v2df)__b, (c)); })

#define _mm_cmp_ss(a, b, c) __extension__ ({ \
  __m128 __a = (a); \
  __m128 __b = (b); \
  (__m128)__builtin_ia32_cmpss((__v4sf)__a, (__v4sf)__b, (c)); })

/* Vector extract */
#define _mm256_extractf128_pd(A, O) __extension__ ({ \
  __m256d __A = (A); \
  (__m128d)__builtin_ia32_vextractf128_pd256((__v4df)__A, (O)); })

#define _mm256_extractf128_ps(A, O) __extension__ ({ \
  __m256 __A = (A); \
  (__m128)__builtin_ia32_vextractf128_ps256((__v8sf)__A, (O)); })

#define _mm256_extractf128_si256(A, O) __extension__ ({ \
  __m256i __A = (A); \
  (__m128i)__builtin_ia32_vextractf128_si256((__v8si)__A, (O)); })

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_extract_epi32(__m256i a, int const imm)
{
  __v8si b = (__v8si)a;
  return b[imm];
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_extract_epi16(__m256i a, int const imm)
{
  __v16hi b = (__v16hi)a;
  return b[imm];
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_extract_epi8(__m256i a, int const imm)
{
  __v32qi b = (__v32qi)a;
  return b[imm];
}

#ifdef __x86_64__
static __inline long long  __attribute__((__always_inline__, __nodebug__))
_mm256_extract_epi64(__m256i a, const int imm)
{
  __v4di b = (__v4di)a;
  return b[imm];
}
#endif

/* Vector insert */
#define _mm256_insertf128_pd(V1, V2, O) __extension__ ({ \
  __m256d __V1 = (V1); \
  __m128d __V2 = (V2); \
  (__m256d)__builtin_ia32_vinsertf128_pd256((__v4df)__V1, (__v2df)__V2, (O)); })

#define _mm256_insertf128_ps(V1, V2, O) __extension__ ({ \
  __m256 __V1 = (V1); \
  __m128 __V2 = (V2); \
  (__m256)__builtin_ia32_vinsertf128_ps256((__v8sf)__V1, (__v4sf)__V2, (O)); })

#define _mm256_insertf128_si256(V1, V2, O) __extension__ ({ \
  __m256i __V1 = (V1); \
  __m128i __V2 = (V2); \
  (__m256i)__builtin_ia32_vinsertf128_si256((__v8si)__V1, (__v4si)__V2, (O)); })

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_insert_epi32(__m256i a, int b, int const imm)
{
  __v8si c = (__v8si)a;
  c[imm & 7] = b;
  return (__m256i)c;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_insert_epi16(__m256i a, int b, int const imm)
{
  __v16hi c = (__v16hi)a;
  c[imm & 15] = b;
  return (__m256i)c;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_insert_epi8(__m256i a, int b, int const imm)
{
  __v32qi c = (__v32qi)a;
  c[imm & 31] = b;
  return (__m256i)c;
}

#ifdef __x86_64__
static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_insert_epi64(__m256i a, int b, int const imm)
{
  __v4di c = (__v4di)a;
  c[imm & 3] = b;
  return (__m256i)c;
}
#endif

/* Conversion */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi32_pd(__m128i a)
{
  return (__m256d)__builtin_ia32_cvtdq2pd256((__v4si) a);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi32_ps(__m256i a)
{
  return (__m256)__builtin_ia32_cvtdq2ps256((__v8si) a);
}

static __inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm256_cvtpd_ps(__m256d a)
{
  return (__m128)__builtin_ia32_cvtpd2ps256((__v4df) a);
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtps_epi32(__m256 a)
{
  return (__m256i)__builtin_ia32_cvtps2dq256((__v8sf) a);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_cvtps_pd(__m128 a)
{
  return (__m256d)__builtin_ia32_cvtps2pd256((__v4sf) a);
}

static __inline __m128i __attribute__((__always_inline__, __nodebug__))
_mm256_cvttpd_epi32(__m256d a)
{
  return (__m128i)__builtin_ia32_cvttpd2dq256((__v4df) a);
}

static __inline __m128i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtpd_epi32(__m256d a)
{
  return (__m128i)__builtin_ia32_cvtpd2dq256((__v4df) a);
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvttps_epi32(__m256 a)
{
  return (__m256i)__builtin_ia32_cvttps2dq256((__v8sf) a);
}

/* Vector replicate */
static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_movehdup_ps(__m256 a)
{
  return __builtin_shufflevector(a, a, 1, 1, 3, 3, 5, 5, 7, 7);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_moveldup_ps(__m256 a)
{
  return __builtin_shufflevector(a, a, 0, 0, 2, 2, 4, 4, 6, 6);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_movedup_pd(__m256d a)
{
  return __builtin_shufflevector(a, a, 0, 0, 2, 2);
}

/* Unpack and Interleave */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_unpackhi_pd(__m256d a, __m256d b)
{
  return __builtin_shufflevector(a, b, 1, 5, 1+2, 5+2);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_unpacklo_pd(__m256d a, __m256d b)
{
  return __builtin_shufflevector(a, b, 0, 4, 0+2, 4+2);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_unpackhi_ps(__m256 a, __m256 b)
{
  return __builtin_shufflevector(a, b, 2, 10, 2+1, 10+1, 6, 14, 6+1, 14+1);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_unpacklo_ps(__m256 a, __m256 b)
{
  return __builtin_shufflevector(a, b, 0, 8, 0+1, 8+1, 4, 12, 4+1, 12+1);
}

/* Bit Test */
static __inline int __attribute__((__always_inline__, __nodebug__))
_mm_testz_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_vtestzpd((__v2df)a, (__v2df)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm_testc_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_vtestcpd((__v2df)a, (__v2df)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm_testnzc_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_vtestnzcpd((__v2df)a, (__v2df)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm_testz_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_vtestzps((__v4sf)a, (__v4sf)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm_testc_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_vtestcps((__v4sf)a, (__v4sf)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm_testnzc_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_vtestnzcps((__v4sf)a, (__v4sf)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testz_pd(__m256d a, __m256d b)
{
  return __builtin_ia32_vtestzpd256((__v4df)a, (__v4df)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testc_pd(__m256d a, __m256d b)
{
  return __builtin_ia32_vtestcpd256((__v4df)a, (__v4df)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testnzc_pd(__m256d a, __m256d b)
{
  return __builtin_ia32_vtestnzcpd256((__v4df)a, (__v4df)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testz_ps(__m256 a, __m256 b)
{
  return __builtin_ia32_vtestzps256((__v8sf)a, (__v8sf)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testc_ps(__m256 a, __m256 b)
{
  return __builtin_ia32_vtestcps256((__v8sf)a, (__v8sf)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testnzc_ps(__m256 a, __m256 b)
{
  return __builtin_ia32_vtestnzcps256((__v8sf)a, (__v8sf)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testz_si256(__m256i a, __m256i b)
{
  return __builtin_ia32_ptestz256((__v4di)a, (__v4di)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testc_si256(__m256i a, __m256i b)
{
  return __builtin_ia32_ptestc256((__v4di)a, (__v4di)b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testnzc_si256(__m256i a, __m256i b)
{
  return __builtin_ia32_ptestnzc256((__v4di)a, (__v4di)b);
}

/* Vector extract sign mask */
static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_movemask_pd(__m256d a)
{
  return __builtin_ia32_movmskpd256((__v4df)a);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_movemask_ps(__m256 a)
{
  return __builtin_ia32_movmskps256((__v8sf)a);
}

/* Vector zero */
static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_zeroall(void)
{
  __builtin_ia32_vzeroall();
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_zeroupper(void)
{
  __builtin_ia32_vzeroupper();
}

/* Vector load with broadcast */
static __inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_broadcast_ss(float const *a)
{
  return (__m128)__builtin_ia32_vbroadcastss(a);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_broadcast_sd(double const *a)
{
  return (__m256d)__builtin_ia32_vbroadcastsd256(a);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_broadcast_ss(float const *a)
{
  return (__m256)__builtin_ia32_vbroadcastss256(a);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_broadcast_pd(__m128d const *a)
{
  return (__m256d)__builtin_ia32_vbroadcastf128_pd256(a);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_broadcast_ps(__m128 const *a)
{
  return (__m256)__builtin_ia32_vbroadcastf128_ps256(a);
}

/* SIMD load ops */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_load_pd(double const *p)
{
  return *(__m256d *)p;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_load_ps(float const *p)
{
  return *(__m256 *)p;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_loadu_pd(double const *p)
{
  struct __loadu_pd {
    __m256d v;
  } __attribute__((packed, may_alias));
  return ((struct __loadu_pd*)p)->v;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_loadu_ps(float const *p)
{
  struct __loadu_ps {
    __m256 v;
  } __attribute__((packed, may_alias));
  return ((struct __loadu_ps*)p)->v;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_load_si256(__m256i const *p)
{
  return *p;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_loadu_si256(__m256i const *p)
{
  struct __loadu_si256 {
    __m256i v;
  } __attribute__((packed, may_alias));
  return ((struct __loadu_si256*)p)->v;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_lddqu_si256(__m256i const *p)
{
  return (__m256i)__builtin_ia32_lddqu256((char const *)p);
}

/* SIMD store ops */
static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_store_pd(double *p, __m256d a)
{
  *(__m256d *)p = a;
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_store_ps(float *p, __m256 a)
{
  *(__m256 *)p = a;
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_storeu_pd(double *p, __m256d a)
{
  __builtin_ia32_storeupd256(p, (__v4df)a);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_storeu_ps(float *p, __m256 a)
{
  __builtin_ia32_storeups256(p, (__v8sf)a);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_store_si256(__m256i *p, __m256i a)
{
  *p = a;
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_storeu_si256(__m256i *p, __m256i a)
{
  __builtin_ia32_storedqu256((char *)p, (__v32qi)a);
}

/* Conditional load ops */
static __inline __m128d __attribute__((__always_inline__, __nodebug__))
_mm_maskload_pd(double const *p, __m128d m)
{
  return (__m128d)__builtin_ia32_maskloadpd((const __v2df *)p, (__v2df)m);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_maskload_pd(double const *p, __m256d m)
{
  return (__m256d)__builtin_ia32_maskloadpd256((const __v4df *)p, (__v4df)m);
}

static __inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_maskload_ps(float const *p, __m128 m)
{
  return (__m128)__builtin_ia32_maskloadps((const __v4sf *)p, (__v4sf)m);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_maskload_ps(float const *p, __m256 m)
{
  return (__m256)__builtin_ia32_maskloadps256((const __v8sf *)p, (__v8sf)m);
}

/* Conditional store ops */
static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_maskstore_ps(float *p, __m256 m, __m256 a)
{
  __builtin_ia32_maskstoreps256((__v8sf *)p, (__v8sf)m, (__v8sf)a);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm_maskstore_pd(double *p, __m128d m, __m128d a)
{
  __builtin_ia32_maskstorepd((__v2df *)p, (__v2df)m, (__v2df)a);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_maskstore_pd(double *p, __m256d m, __m256d a)
{
  __builtin_ia32_maskstorepd256((__v4df *)p, (__v4df)m, (__v4df)a);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm_maskstore_ps(float *p, __m128 m, __m128 a)
{
  __builtin_ia32_maskstoreps((__v4sf *)p, (__v4sf)m, (__v4sf)a);
}

/* Cacheability support ops */
static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_stream_si256(__m256i *a, __m256i b)
{
  __builtin_ia32_movntdq256((__v4di *)a, (__v4di)b);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_stream_pd(double *a, __m256d b)
{
  __builtin_ia32_movntpd256(a, (__v4df)b);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_stream_ps(float *p, __m256 a)
{
  __builtin_ia32_movntps256(p, (__v8sf)a);
}

/* Create vectors */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_set_pd(double a, double b, double c, double d)
{
  return (__m256d){ d, c, b, a };
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_set_ps(float a, float b, float c, float d,
	            float e, float f, float g, float h)
{
  return (__m256){ h, g, f, e, d, c, b, a };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set_epi32(int i0, int i1, int i2, int i3,
		             int i4, int i5, int i6, int i7)
{
  return (__m256i)(__v8si){ i7, i6, i5, i4, i3, i2, i1, i0 };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set_epi16(short w15, short w14, short w13, short w12,
		             short w11, short w10, short w09, short w08,
		             short w07, short w06, short w05, short w04,
		             short w03, short w02, short w01, short w00)
{
  return (__m256i)(__v16hi){ w00, w01, w02, w03, w04, w05, w06, w07,
                             w08, w09, w10, w11, w12, w13, w14, w15 };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set_epi8(char b31, char b30, char b29, char b28,
		            char b27, char b26, char b25, char b24,
		            char b23, char b22, char b21, char b20,
		            char b19, char b18, char b17, char b16,
		            char b15, char b14, char b13, char b12,
		            char b11, char b10, char b09, char b08,
		            char b07, char b06, char b05, char b04,
		            char b03, char b02, char b01, char b00)
{
  return (__m256i)(__v32qi){
    b00, b01, b02, b03, b04, b05, b06, b07,
    b08, b09, b10, b11, b12, b13, b14, b15,
    b16, b17, b18, b19, b20, b21, b22, b23,
    b24, b25, b26, b27, b28, b29, b30, b31
  };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set_epi64x(long long a, long long b, long long c, long long d)
{
  return (__m256i)(__v4di){ d, c, b, a };
}

/* Create vectors with elements in reverse order */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_setr_pd(double a, double b, double c, double d)
{
  return (__m256d){ a, b, c, d };
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_setr_ps(float a, float b, float c, float d,
		           float e, float f, float g, float h)
{
  return (__m256){ a, b, c, d, e, f, g, h };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_setr_epi32(int i0, int i1, int i2, int i3,
		              int i4, int i5, int i6, int i7)
{
  return (__m256i)(__v8si){ i0, i1, i2, i3, i4, i5, i6, i7 };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_setr_epi16(short w15, short w14, short w13, short w12,
		   short w11, short w10, short w09, short w08,
		   short w07, short w06, short w05, short w04,
		   short w03, short w02, short w01, short w00)
{
  return (__m256i)(__v16hi){ w15, w14, w13, w12, w11, w10, w09, w08,
			                       w07, w06, w05, w04, w03, w02, w01, w00 };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_setr_epi8(char b31, char b30, char b29, char b28,
		             char b27, char b26, char b25, char b24,
		             char b23, char b22, char b21, char b20,
		             char b19, char b18, char b17, char b16,
		             char b15, char b14, char b13, char b12,
		             char b11, char b10, char b09, char b08,
		             char b07, char b06, char b05, char b04,
		             char b03, char b02, char b01, char b00)
{
  return (__m256i)(__v32qi){
    b31, b30, b29, b28, b27, b26, b25, b24,
		b23, b22, b21, b20, b19, b18, b17, b16,
		b15, b14, b13, b12, b11, b10, b09, b08,
		b07, b06, b05, b04, b03, b02, b01, b00 };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_setr_epi64x(long long a, long long b, long long c, long long d)
{
  return (__m256i)(__v4di){ a, b, c, d };
}

/* Create vectors with repeated elements */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_set1_pd(double w)
{
  return (__m256d){ w, w, w, w };
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_set1_ps(float w)
{
  return (__m256){ w, w, w, w, w, w, w, w };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set1_epi32(int i)
{
  return (__m256i)(__v8si){ i, i, i, i, i, i, i, i };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set1_epi16(short w)
{
  return (__m256i)(__v16hi){ w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set1_epi8(char b)
{
  return (__m256i)(__v32qi){ b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b,
                             b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set1_epi64x(long long q)
{
  return (__m256i)(__v4di){ q, q, q, q };
}

/* Create zeroed vectors */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_setzero_pd(void)
{
  return (__m256d){ 0, 0, 0, 0 };
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_setzero_ps(void)
{
  return (__m256){ 0, 0, 0, 0, 0, 0, 0, 0 };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_setzero_si256(void)
{
  return (__m256i){ 0LL, 0LL, 0LL, 0LL };
}

/* Cast between vector types */
static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_castpd_ps(__m256d in)
{
  return (__m256)in;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_castpd_si256(__m256d in)
{
  return (__m256i)in;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_castps_pd(__m256 in)
{
  return (__m256d)in;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_castps_si256(__m256 in)
{
  return (__m256i)in;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_castsi256_ps(__m256i in)
{
  return (__m256)in;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_castsi256_pd(__m256i in)
{
  return (__m256d)in;
}

static __inline __m128d __attribute__((__always_inline__, __nodebug__))
_mm256_castpd256_pd128(__m256d in)
{
  return __builtin_shufflevector(in, in, 0, 1);
}

static __inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm256_castps256_ps128(__m256 in)
{
  return __builtin_shufflevector(in, in, 0, 1, 2, 3);
}

static __inline __m128i __attribute__((__always_inline__, __nodebug__))
_mm256_castsi256_si128(__m256i in)
{
  return __builtin_shufflevector(in, in, 0, 1);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_castpd128_pd256(__m128d in)
{
  __m128d zero = _mm_setzero_pd();
  return __builtin_shufflevector(in, zero, 0, 1, 2, 2);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_castps128_ps256(__m128 in)
{
  __m128 zero = _mm_setzero_ps();
  return __builtin_shufflevector(in, zero, 0, 1, 2, 3, 4, 4, 4, 4);
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_castsi128_si256(__m128i in)
{
  __m128i zero = _mm_setzero_si128();
  return __builtin_shufflevector(in, zero, 0, 1, 2, 2);
}

/* SIMD load ops (unaligned) */
static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_loadu2_m128(float const *addr_hi, float const *addr_lo)
{
  struct __loadu_ps {
    __m128 v;
  } __attribute__((__packed__, __may_alias__));

  __m256 v256 = _mm256_castps128_ps256(((struct __loadu_ps*)addr_lo)->v);
  return _mm256_insertf128_ps(v256, ((struct __loadu_ps*)addr_hi)->v, 1);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_loadu2_m128d(double const *addr_hi, double const *addr_lo)
{
  struct __loadu_pd {
    __m128d v;
  } __attribute__((__packed__, __may_alias__));
  
  __m256d v256 = _mm256_castpd128_pd256(((struct __loadu_pd*)addr_lo)->v);
  return _mm256_insertf128_pd(v256, ((struct __loadu_pd*)addr_hi)->v, 1);
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_loadu2_m128i(__m128i const *addr_hi, __m128i const *addr_lo)
{
  struct __loadu_si128 {
    __m128i v;
  } __attribute__((packed, may_alias));
  __m256i v256 = _mm256_castsi128_si256(((struct __loadu_si128*)addr_lo)->v);
  return _mm256_insertf128_si256(v256, ((struct __loadu_si128*)addr_hi)->v, 1);
}

/* SIMD store ops (unaligned) */
static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_storeu2_m128(float *addr_hi, float *addr_lo, __m256 a)
{
  __m128 v128;

  v128 = _mm256_castps256_ps128(a);
  __builtin_ia32_storeups(addr_lo, v128);
  v128 = _mm256_extractf128_ps(a, 1);
  __builtin_ia32_storeups(addr_hi, v128);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_storeu2_m128d(double *addr_hi, double *addr_lo, __m256d a)
{
  __m128d v128;

  v128 = _mm256_castpd256_pd128(a);
  __builtin_ia32_storeupd(addr_lo, v128);
  v128 = _mm256_extractf128_pd(a, 1);
  __builtin_ia32_storeupd(addr_hi, v128);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_storeu2_m128i(__m128i *addr_hi, __m128i *addr_lo, __m256i a)
{
  __m128i v128;

  v128 = _mm256_castsi256_si128(a);
  __builtin_ia32_storedqu((char *)addr_lo, (__v16qi)v128);
  v128 = _mm256_extractf128_si256(a, 1);
  __builtin_ia32_storedqu((char *)addr_hi, (__v16qi)v128);
}
