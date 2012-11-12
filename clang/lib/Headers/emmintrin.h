/*===---- emmintrin.h - SSE2 intrinsics ------------------------------------===
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

#ifndef __EMMINTRIN_H
#define __EMMINTRIN_H

#ifndef __SSE2__
#error "SSE2 instruction set not enabled"
#else

#include <xmmintrin.h>

typedef double __m128d __attribute__((__vector_size__(16)));
typedef long long __m128i __attribute__((__vector_size__(16)));

/* Type defines.  */
typedef double __v2df __attribute__ ((__vector_size__ (16)));
typedef long long __v2di __attribute__ ((__vector_size__ (16)));
typedef short __v8hi __attribute__((__vector_size__(16)));
typedef char __v16qi __attribute__((__vector_size__(16)));

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_add_sd(__m128d a, __m128d b)
{
  a[0] += b[0];
  return a;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_add_pd(__m128d a, __m128d b)
{
  return a + b;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_sub_sd(__m128d a, __m128d b)
{
  a[0] -= b[0];
  return a;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_sub_pd(__m128d a, __m128d b)
{
  return a - b;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_mul_sd(__m128d a, __m128d b)
{
  a[0] *= b[0];
  return a;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_mul_pd(__m128d a, __m128d b)
{
  return a * b;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_div_sd(__m128d a, __m128d b)
{
  a[0] /= b[0];
  return a;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_div_pd(__m128d a, __m128d b)
{
  return a / b;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_sqrt_sd(__m128d a, __m128d b)
{
  __m128d c = __builtin_ia32_sqrtsd(b);
  return (__m128d) { c[0], a[1] };
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_sqrt_pd(__m128d a)
{
  return __builtin_ia32_sqrtpd(a);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_min_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_minsd(a, b);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_min_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_minpd(a, b);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_max_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_maxsd(a, b);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_max_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_maxpd(a, b);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_and_pd(__m128d a, __m128d b)
{
  return (__m128d)((__v4si)a & (__v4si)b);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_andnot_pd(__m128d a, __m128d b)
{
  return (__m128d)(~(__v4si)a & (__v4si)b);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_or_pd(__m128d a, __m128d b)
{
  return (__m128d)((__v4si)a | (__v4si)b);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_xor_pd(__m128d a, __m128d b)
{
  return (__m128d)((__v4si)a ^ (__v4si)b);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmppd(a, b, 0);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmppd(a, b, 1);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmple_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmppd(a, b, 2);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmppd(b, a, 1);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmppd(b, a, 2);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpord_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmppd(a, b, 7);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpunord_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmppd(a, b, 3);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmppd(a, b, 4);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpnlt_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmppd(a, b, 5);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpnle_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmppd(a, b, 6);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpngt_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmppd(b, a, 5);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpnge_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmppd(b, a, 6);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpsd(a, b, 0);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpsd(a, b, 1);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmple_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpsd(a, b, 2);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpsd(b, a, 1);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpsd(b, a, 2);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpord_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpsd(a, b, 7);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpunord_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpsd(a, b, 3);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpsd(a, b, 4);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpnlt_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpsd(a, b, 5);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpnle_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpsd(a, b, 6);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpngt_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpsd(b, a, 5);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cmpnge_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpsd(b, a, 6);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_comieq_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_comisdeq(a, b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_comilt_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_comisdlt(a, b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_comile_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_comisdle(a, b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_comigt_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_comisdgt(a, b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_comige_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_comisdge(a, b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_comineq_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_comisdneq(a, b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_ucomieq_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_ucomisdeq(a, b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_ucomilt_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_ucomisdlt(a, b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_ucomile_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_ucomisdle(a, b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_ucomigt_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_ucomisdgt(a, b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_ucomige_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_ucomisdge(a, b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_ucomineq_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_ucomisdneq(a, b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpd_ps(__m128d a)
{
  return __builtin_ia32_cvtpd2ps(a);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cvtps_pd(__m128 a)
{
  return __builtin_ia32_cvtps2pd(a);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cvtepi32_pd(__m128i a)
{
  return __builtin_ia32_cvtdq2pd((__v4si)a);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cvtpd_epi32(__m128d a)
{
  return __builtin_ia32_cvtpd2dq(a);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_cvtsd_si32(__m128d a)
{
  return __builtin_ia32_cvtsd2si(a);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtsd_ss(__m128 a, __m128d b)
{
  a[0] = b[0];
  return a;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cvtsi32_sd(__m128d a, int b)
{
  a[0] = b;
  return a;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cvtss_sd(__m128d a, __m128 b)
{
  a[0] = b[0];
  return a;
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cvttpd_epi32(__m128d a)
{
  return (__m128i)__builtin_ia32_cvttpd2dq(a);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_cvttsd_si32(__m128d a)
{
  return a[0];
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpd_pi32(__m128d a)
{
  return (__m64)__builtin_ia32_cvtpd2pi(a);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvttpd_pi32(__m128d a)
{
  return (__m64)__builtin_ia32_cvttpd2pi(a);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cvtpi32_pd(__m64 a)
{
  return __builtin_ia32_cvtpi2pd((__v2si)a);
}

static __inline__ double __attribute__((__always_inline__, __nodebug__))
_mm_cvtsd_f64(__m128d a)
{
  return a[0];
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_load_pd(double const *dp)
{
  return *(__m128d*)dp;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_load1_pd(double const *dp)
{
  struct __mm_load1_pd_struct {
    double u;
  } __attribute__((__packed__, __may_alias__));
  double u = ((struct __mm_load1_pd_struct*)dp)->u;
  return (__m128d){ u, u };
}

#define        _mm_load_pd1(dp)        _mm_load1_pd(dp)

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_loadr_pd(double const *dp)
{
  __m128d u = *(__m128d*)dp;
  return __builtin_shufflevector(u, u, 1, 0);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_loadu_pd(double const *dp)
{
  struct __loadu_pd {
    __m128d v;
  } __attribute__((packed, may_alias));
  return ((struct __loadu_pd*)dp)->v;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_load_sd(double const *dp)
{
  struct __mm_load_sd_struct {
    double u;
  } __attribute__((__packed__, __may_alias__));
  double u = ((struct __mm_load_sd_struct*)dp)->u;
  return (__m128d){ u, 0 };
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_loadh_pd(__m128d a, double const *dp)
{
  struct __mm_loadh_pd_struct {
    double u;
  } __attribute__((__packed__, __may_alias__));
  double u = ((struct __mm_loadh_pd_struct*)dp)->u;
  return (__m128d){ a[0], u };
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_loadl_pd(__m128d a, double const *dp)
{
  struct __mm_loadl_pd_struct {
    double u;
  } __attribute__((__packed__, __may_alias__));
  double u = ((struct __mm_loadl_pd_struct*)dp)->u;
  return (__m128d){ u, a[1] }; 
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_set_sd(double w)
{
  return (__m128d){ w, 0 };
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_set1_pd(double w)
{
  return (__m128d){ w, w };
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_set_pd(double w, double x)
{
  return (__m128d){ x, w };
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_setr_pd(double w, double x)
{
  return (__m128d){ w, x };
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_setzero_pd(void)
{
  return (__m128d){ 0, 0 };
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_move_sd(__m128d a, __m128d b)
{
  return (__m128d){ b[0], a[1] };
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_store_sd(double *dp, __m128d a)
{
  struct __mm_store_sd_struct {
    double u;
  } __attribute__((__packed__, __may_alias__));
  ((struct __mm_store_sd_struct*)dp)->u = a[0];
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_store1_pd(double *dp, __m128d a)
{
  struct __mm_store1_pd_struct {
    double u[2];
  } __attribute__((__packed__, __may_alias__));
  ((struct __mm_store1_pd_struct*)dp)->u[0] = a[0];
  ((struct __mm_store1_pd_struct*)dp)->u[1] = a[0];
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_store_pd(double *dp, __m128d a)
{
  *(__m128d *)dp = a;
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_storeu_pd(double *dp, __m128d a)
{
  __builtin_ia32_storeupd(dp, a);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_storer_pd(double *dp, __m128d a)
{
  a = __builtin_shufflevector(a, a, 1, 0);
  *(__m128d *)dp = a;
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_storeh_pd(double *dp, __m128d a)
{
  struct __mm_storeh_pd_struct {
    double u;
  } __attribute__((__packed__, __may_alias__));
  ((struct __mm_storeh_pd_struct*)dp)->u = a[1];
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_storel_pd(double *dp, __m128d a)
{
  struct __mm_storeh_pd_struct {
    double u;
  } __attribute__((__packed__, __may_alias__));
  ((struct __mm_storeh_pd_struct*)dp)->u = a[0];
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_add_epi8(__m128i a, __m128i b)
{
  return (__m128i)((__v16qi)a + (__v16qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_add_epi16(__m128i a, __m128i b)
{
  return (__m128i)((__v8hi)a + (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_add_epi32(__m128i a, __m128i b)
{
  return (__m128i)((__v4si)a + (__v4si)b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_add_si64(__m64 a, __m64 b)
{
  return a + b;
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_add_epi64(__m128i a, __m128i b)
{
  return a + b;
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_adds_epi8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_paddsb128((__v16qi)a, (__v16qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_adds_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_paddsw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_adds_epu8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_paddusb128((__v16qi)a, (__v16qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_adds_epu16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_paddusw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_avg_epu8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pavgb128((__v16qi)a, (__v16qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_avg_epu16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pavgw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_madd_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pmaddwd128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_max_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pmaxsw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_max_epu8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pmaxub128((__v16qi)a, (__v16qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_min_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pminsw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_min_epu8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pminub128((__v16qi)a, (__v16qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mulhi_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pmulhw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mulhi_epu16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pmulhuw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mullo_epi16(__m128i a, __m128i b)
{
  return (__m128i)((__v8hi)a * (__v8hi)b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_mul_su32(__m64 a, __m64 b)
{
  return __builtin_ia32_pmuludq((__v2si)a, (__v2si)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mul_epu32(__m128i a, __m128i b)
{
  return __builtin_ia32_pmuludq128((__v4si)a, (__v4si)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sad_epu8(__m128i a, __m128i b)
{
  return __builtin_ia32_psadbw128((__v16qi)a, (__v16qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sub_epi8(__m128i a, __m128i b)
{
  return (__m128i)((__v16qi)a - (__v16qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sub_epi16(__m128i a, __m128i b)
{
  return (__m128i)((__v8hi)a - (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sub_epi32(__m128i a, __m128i b)
{
  return (__m128i)((__v4si)a - (__v4si)b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sub_si64(__m64 a, __m64 b)
{
  return a - b;
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sub_epi64(__m128i a, __m128i b)
{
  return a - b;
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_subs_epi8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_psubsb128((__v16qi)a, (__v16qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_subs_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_psubsw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_subs_epu8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_psubusb128((__v16qi)a, (__v16qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_subs_epu16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_psubusw128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_and_si128(__m128i a, __m128i b)
{
  return a & b;
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_andnot_si128(__m128i a, __m128i b)
{
  return ~a & b;
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_or_si128(__m128i a, __m128i b)
{
  return a | b;
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_xor_si128(__m128i a, __m128i b)
{
  return a ^ b;
}

#define _mm_slli_si128(a, count) __extension__ ({ \
  __m128i __a = (a); \
  (__m128i)__builtin_ia32_pslldqi128(__a, (count)*8); })

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_slli_epi16(__m128i a, int count)
{
  return (__m128i)__builtin_ia32_psllwi128((__v8hi)a, count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sll_epi16(__m128i a, __m128i count)
{
  return (__m128i)__builtin_ia32_psllw128((__v8hi)a, (__v8hi)count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_slli_epi32(__m128i a, int count)
{
  return (__m128i)__builtin_ia32_pslldi128((__v4si)a, count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sll_epi32(__m128i a, __m128i count)
{
  return (__m128i)__builtin_ia32_pslld128((__v4si)a, (__v4si)count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_slli_epi64(__m128i a, int count)
{
  return __builtin_ia32_psllqi128(a, count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sll_epi64(__m128i a, __m128i count)
{
  return __builtin_ia32_psllq128(a, count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_srai_epi16(__m128i a, int count)
{
  return (__m128i)__builtin_ia32_psrawi128((__v8hi)a, count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sra_epi16(__m128i a, __m128i count)
{
  return (__m128i)__builtin_ia32_psraw128((__v8hi)a, (__v8hi)count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_srai_epi32(__m128i a, int count)
{
  return (__m128i)__builtin_ia32_psradi128((__v4si)a, count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sra_epi32(__m128i a, __m128i count)
{
  return (__m128i)__builtin_ia32_psrad128((__v4si)a, (__v4si)count);
}


#define _mm_srli_si128(a, count) __extension__ ({ \
  __m128i __a = (a); \
  (__m128i)__builtin_ia32_psrldqi128(__a, (count)*8); })

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_srli_epi16(__m128i a, int count)
{
  return (__m128i)__builtin_ia32_psrlwi128((__v8hi)a, count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_srl_epi16(__m128i a, __m128i count)
{
  return (__m128i)__builtin_ia32_psrlw128((__v8hi)a, (__v8hi)count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_srli_epi32(__m128i a, int count)
{
  return (__m128i)__builtin_ia32_psrldi128((__v4si)a, count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_srl_epi32(__m128i a, __m128i count)
{
  return (__m128i)__builtin_ia32_psrld128((__v4si)a, (__v4si)count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_srli_epi64(__m128i a, int count)
{
  return __builtin_ia32_psrlqi128(a, count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_srl_epi64(__m128i a, __m128i count)
{
  return __builtin_ia32_psrlq128(a, count);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_epi8(__m128i a, __m128i b)
{
  return (__m128i)((__v16qi)a == (__v16qi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_epi16(__m128i a, __m128i b)
{
  return (__m128i)((__v8hi)a == (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_epi32(__m128i a, __m128i b)
{
  return (__m128i)((__v4si)a == (__v4si)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_epi8(__m128i a, __m128i b)
{
  /* This function always performs a signed comparison, but __v16qi is a char
     which may be signed or unsigned. */
  typedef signed char __v16qs __attribute__((__vector_size__(16)));
  return (__m128i)((__v16qs)a > (__v16qs)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_epi16(__m128i a, __m128i b)
{
  return (__m128i)((__v8hi)a > (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_epi32(__m128i a, __m128i b)
{
  return (__m128i)((__v4si)a > (__v4si)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_epi8(__m128i a, __m128i b)
{
  return _mm_cmpgt_epi8(b,a);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_epi16(__m128i a, __m128i b)
{
  return _mm_cmpgt_epi16(b,a);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_epi32(__m128i a, __m128i b)
{
  return _mm_cmpgt_epi32(b,a);
}

#ifdef __x86_64__
static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_cvtsi64_sd(__m128d a, long long b)
{
  a[0] = b;
  return a;
}

static __inline__ long long __attribute__((__always_inline__, __nodebug__))
_mm_cvtsd_si64(__m128d a)
{
  return __builtin_ia32_cvtsd2si64(a);
}

static __inline__ long long __attribute__((__always_inline__, __nodebug__))
_mm_cvttsd_si64(__m128d a)
{
  return a[0];
}
#endif

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtepi32_ps(__m128i a)
{
  return __builtin_ia32_cvtdq2ps((__v4si)a);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cvtps_epi32(__m128 a)
{
  return (__m128i)__builtin_ia32_cvtps2dq(a);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cvttps_epi32(__m128 a)
{
  return (__m128i)__builtin_ia32_cvttps2dq(a);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cvtsi32_si128(int a)
{
  return (__m128i)(__v4si){ a, 0, 0, 0 };
}

#ifdef __x86_64__
static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cvtsi64_si128(long long a)
{
  return (__m128i){ a, 0 };
}
#endif

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_cvtsi128_si32(__m128i a)
{
  __v4si b = (__v4si)a;
  return b[0];
}

#ifdef __x86_64__
static __inline__ long long __attribute__((__always_inline__, __nodebug__))
_mm_cvtsi128_si64(__m128i a)
{
  return a[0];
}
#endif

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_load_si128(__m128i const *p)
{
  return *p;
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_loadu_si128(__m128i const *p)
{
  struct __loadu_si128 {
    __m128i v;
  } __attribute__((packed, may_alias));
  return ((struct __loadu_si128*)p)->v;
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_loadl_epi64(__m128i const *p)
{
  struct __mm_loadl_epi64_struct {
    long long u;
  } __attribute__((__packed__, __may_alias__));
  return (__m128i) { ((struct __mm_loadl_epi64_struct*)p)->u, 0};
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_set_epi64x(long long q1, long long q0)
{
  return (__m128i){ q0, q1 };
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_set_epi64(__m64 q1, __m64 q0)
{
  return (__m128i){ (long long)q0, (long long)q1 };
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_set_epi32(int i3, int i2, int i1, int i0)
{
  return (__m128i)(__v4si){ i0, i1, i2, i3};
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_set_epi16(short w7, short w6, short w5, short w4, short w3, short w2, short w1, short w0)
{
  return (__m128i)(__v8hi){ w0, w1, w2, w3, w4, w5, w6, w7 };
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_set_epi8(char b15, char b14, char b13, char b12, char b11, char b10, char b9, char b8, char b7, char b6, char b5, char b4, char b3, char b2, char b1, char b0)
{
  return (__m128i)(__v16qi){ b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15 };
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_set1_epi64x(long long q)
{
  return (__m128i){ q, q };
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_set1_epi64(__m64 q)
{
  return (__m128i){ (long long)q, (long long)q };
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_set1_epi32(int i)
{
  return (__m128i)(__v4si){ i, i, i, i };
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_set1_epi16(short w)
{
  return (__m128i)(__v8hi){ w, w, w, w, w, w, w, w };
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_set1_epi8(char b)
{
  return (__m128i)(__v16qi){ b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b };
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_setr_epi64(__m64 q0, __m64 q1)
{
  return (__m128i){ (long long)q0, (long long)q1 };
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_setr_epi32(int i0, int i1, int i2, int i3)
{
  return (__m128i)(__v4si){ i0, i1, i2, i3};
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_setr_epi16(short w0, short w1, short w2, short w3, short w4, short w5, short w6, short w7)
{
  return (__m128i)(__v8hi){ w0, w1, w2, w3, w4, w5, w6, w7 };
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_setr_epi8(char b0, char b1, char b2, char b3, char b4, char b5, char b6, char b7, char b8, char b9, char b10, char b11, char b12, char b13, char b14, char b15)
{
  return (__m128i)(__v16qi){ b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15 };
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_setzero_si128(void)
{
  return (__m128i){ 0LL, 0LL };
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_store_si128(__m128i *p, __m128i b)
{
  *p = b;
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_storeu_si128(__m128i *p, __m128i b)
{
  __builtin_ia32_storedqu((char *)p, (__v16qi)b);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_maskmoveu_si128(__m128i d, __m128i n, char *p)
{
  __builtin_ia32_maskmovdqu((__v16qi)d, (__v16qi)n, p);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_storel_epi64(__m128i *p, __m128i a)
{
  struct __mm_storel_epi64_struct {
    long long u;
  } __attribute__((__packed__, __may_alias__));
  ((struct __mm_storel_epi64_struct*)p)->u = a[0];
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_stream_pd(double *p, __m128d a)
{
  __builtin_ia32_movntpd(p, a);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_stream_si128(__m128i *p, __m128i a)
{
  __builtin_ia32_movntdq(p, a);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_stream_si32(int *p, int a)
{
  __builtin_ia32_movnti(p, a);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_clflush(void const *p)
{
  __builtin_ia32_clflush(p);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_lfence(void)
{
  __builtin_ia32_lfence();
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_mfence(void)
{
  __builtin_ia32_mfence();
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_packs_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_packsswb128((__v8hi)a, (__v8hi)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_packs_epi32(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_packssdw128((__v4si)a, (__v4si)b);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_packus_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_packuswb128((__v8hi)a, (__v8hi)b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_extract_epi16(__m128i a, int imm)
{
  __v8hi b = (__v8hi)a;
  return (unsigned short)b[imm];
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_insert_epi16(__m128i a, int b, int imm)
{
  __v8hi c = (__v8hi)a;
  c[imm & 7] = b;
  return (__m128i)c;
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_movemask_epi8(__m128i a)
{
  return __builtin_ia32_pmovmskb128((__v16qi)a);
}

#define _mm_shuffle_epi32(a, imm) __extension__ ({ \
  __m128i __a = (a); \
  (__m128i)__builtin_shufflevector((__v4si)__a, (__v4si) _mm_set1_epi32(0), \
                                   (imm) & 0x3, ((imm) & 0xc) >> 2, \
                                   ((imm) & 0x30) >> 4, ((imm) & 0xc0) >> 6); })

#define _mm_shufflelo_epi16(a, imm) __extension__ ({ \
  __m128i __a = (a); \
  (__m128i)__builtin_shufflevector((__v8hi)__a, (__v8hi) _mm_set1_epi16(0), \
                                   (imm) & 0x3, ((imm) & 0xc) >> 2, \
                                   ((imm) & 0x30) >> 4, ((imm) & 0xc0) >> 6, \
                                   4, 5, 6, 7); })

#define _mm_shufflehi_epi16(a, imm) __extension__ ({ \
  __m128i __a = (a); \
  (__m128i)__builtin_shufflevector((__v8hi)__a, (__v8hi) _mm_set1_epi16(0), \
                                   0, 1, 2, 3, \
                                   4 + (((imm) & 0x03) >> 0), \
                                   4 + (((imm) & 0x0c) >> 2), \
                                   4 + (((imm) & 0x30) >> 4), \
                                   4 + (((imm) & 0xc0) >> 6)); })

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_unpackhi_epi8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_shufflevector((__v16qi)a, (__v16qi)b, 8, 16+8, 9, 16+9, 10, 16+10, 11, 16+11, 12, 16+12, 13, 16+13, 14, 16+14, 15, 16+15);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_unpackhi_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_shufflevector((__v8hi)a, (__v8hi)b, 4, 8+4, 5, 8+5, 6, 8+6, 7, 8+7);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_unpackhi_epi32(__m128i a, __m128i b)
{
  return (__m128i)__builtin_shufflevector((__v4si)a, (__v4si)b, 2, 4+2, 3, 4+3);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_unpackhi_epi64(__m128i a, __m128i b)
{
  return (__m128i)__builtin_shufflevector(a, b, 1, 2+1);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_unpacklo_epi8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_shufflevector((__v16qi)a, (__v16qi)b, 0, 16+0, 1, 16+1, 2, 16+2, 3, 16+3, 4, 16+4, 5, 16+5, 6, 16+6, 7, 16+7);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_unpacklo_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_shufflevector((__v8hi)a, (__v8hi)b, 0, 8+0, 1, 8+1, 2, 8+2, 3, 8+3);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_unpacklo_epi32(__m128i a, __m128i b)
{
  return (__m128i)__builtin_shufflevector((__v4si)a, (__v4si)b, 0, 4+0, 1, 4+1);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_unpacklo_epi64(__m128i a, __m128i b)
{
  return (__m128i)__builtin_shufflevector(a, b, 0, 2+0);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_movepi64_pi64(__m128i a)
{
  return (__m64)a[0];
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_movpi64_pi64(__m64 a)
{
  return (__m128i){ (long long)a, 0 };
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_move_epi64(__m128i a)
{
  return __builtin_shufflevector(a, (__m128i){ 0 }, 0, 2);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_unpackhi_pd(__m128d a, __m128d b)
{
  return __builtin_shufflevector(a, b, 1, 2+1);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_unpacklo_pd(__m128d a, __m128d b)
{
  return __builtin_shufflevector(a, b, 0, 2+0);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_movemask_pd(__m128d a)
{
  return __builtin_ia32_movmskpd(a);
}

#define _mm_shuffle_pd(a, b, i) __extension__ ({ \
  __m128d __a = (a); \
  __m128d __b = (b); \
  __builtin_shufflevector(__a, __b, (i) & 1, (((i) & 2) >> 1) + 2); })

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_castpd_ps(__m128d in)
{
  return (__m128)in;
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_castpd_si128(__m128d in)
{
  return (__m128i)in;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_castps_pd(__m128 in)
{
  return (__m128d)in;
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_castps_si128(__m128 in)
{
  return (__m128i)in;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_castsi128_ps(__m128i in)
{
  return (__m128)in;
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_castsi128_pd(__m128i in)
{
  return (__m128d)in;
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_pause(void)
{
  __asm__ volatile ("pause");
}

#define _MM_SHUFFLE2(x, y) (((x) << 1) | (y))

#endif /* __SSE2__ */

#endif /* __EMMINTRIN_H */
