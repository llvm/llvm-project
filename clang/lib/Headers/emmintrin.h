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

#include <xmmintrin.h>

typedef double __m128d __attribute__((__vector_size__(16)));
typedef long long __m128i __attribute__((__vector_size__(16)));

/* Type defines.  */
typedef double __v2df __attribute__ ((__vector_size__ (16)));
typedef long long __v2di __attribute__ ((__vector_size__ (16)));
typedef short __v8hi __attribute__((__vector_size__(16)));
typedef char __v16qi __attribute__((__vector_size__(16)));

/* We need an explicitly signed variant for char. Note that this shouldn't
 * appear in the interface though. */
typedef signed char __v16qs __attribute__((__vector_size__(16)));

#include <f16cintrin.h>

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__, __target__("sse2")))

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_add_sd(__m128d __a, __m128d __b)
{
  __a[0] += __b[0];
  return __a;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_add_pd(__m128d __a, __m128d __b)
{
  return __a + __b;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_sub_sd(__m128d __a, __m128d __b)
{
  __a[0] -= __b[0];
  return __a;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_sub_pd(__m128d __a, __m128d __b)
{
  return __a - __b;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mul_sd(__m128d __a, __m128d __b)
{
  __a[0] *= __b[0];
  return __a;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_mul_pd(__m128d __a, __m128d __b)
{
  return __a * __b;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_div_sd(__m128d __a, __m128d __b)
{
  __a[0] /= __b[0];
  return __a;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_div_pd(__m128d __a, __m128d __b)
{
  return __a / __b;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_sqrt_sd(__m128d __a, __m128d __b)
{
  __m128d __c = __builtin_ia32_sqrtsd(__b);
  return (__m128d) { __c[0], __a[1] };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_sqrt_pd(__m128d __a)
{
  return __builtin_ia32_sqrtpd(__a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_min_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_minsd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_min_pd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_minpd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_max_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_maxsd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_max_pd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_maxpd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_and_pd(__m128d __a, __m128d __b)
{
  return (__m128d)((__v4si)__a & (__v4si)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_andnot_pd(__m128d __a, __m128d __b)
{
  return (__m128d)(~(__v4si)__a & (__v4si)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_or_pd(__m128d __a, __m128d __b)
{
  return (__m128d)((__v4si)__a | (__v4si)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_xor_pd(__m128d __a, __m128d __b)
{
  return (__m128d)((__v4si)__a ^ (__v4si)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpeq_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpeqpd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmplt_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpltpd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmple_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmplepd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpgt_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpltpd(__b, __a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpge_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmplepd(__b, __a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpord_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpordpd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpunord_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpunordpd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpneq_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpneqpd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpnlt_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpnltpd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpnle_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpnlepd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpngt_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpnltpd(__b, __a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpnge_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpnlepd(__b, __a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpeq_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpeqsd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmplt_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpltsd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmple_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmplesd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpgt_sd(__m128d __a, __m128d __b)
{
  __m128d __c = __builtin_ia32_cmpltsd(__b, __a);
  return (__m128d) { __c[0], __a[1] };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpge_sd(__m128d __a, __m128d __b)
{
  __m128d __c = __builtin_ia32_cmplesd(__b, __a);
  return (__m128d) { __c[0], __a[1] };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpord_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpordsd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpunord_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpunordsd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpneq_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpneqsd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpnlt_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpnltsd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpnle_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpnlesd(__a, __b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpngt_sd(__m128d __a, __m128d __b)
{
  __m128d __c = __builtin_ia32_cmpnltsd(__b, __a);
  return (__m128d) { __c[0], __a[1] };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpnge_sd(__m128d __a, __m128d __b)
{
  __m128d __c = __builtin_ia32_cmpnlesd(__b, __a);
  return (__m128d) { __c[0], __a[1] };
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_comieq_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_comisdeq(__a, __b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_comilt_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_comisdlt(__a, __b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_comile_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_comisdle(__a, __b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_comigt_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_comisdgt(__a, __b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_comige_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_comisdge(__a, __b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_comineq_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_comisdneq(__a, __b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomieq_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_ucomisdeq(__a, __b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomilt_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_ucomisdlt(__a, __b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomile_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_ucomisdle(__a, __b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomigt_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_ucomisdgt(__a, __b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomige_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_ucomisdge(__a, __b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomineq_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_ucomisdneq(__a, __b);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtpd_ps(__m128d __a)
{
  return __builtin_ia32_cvtpd2ps(__a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cvtps_pd(__m128 __a)
{
  return __builtin_ia32_cvtps2pd(__a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cvtepi32_pd(__m128i __a)
{
  return __builtin_ia32_cvtdq2pd((__v4si)__a);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvtpd_epi32(__m128d __a)
{
  return __builtin_ia32_cvtpd2dq(__a);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_cvtsd_si32(__m128d __a)
{
  return __builtin_ia32_cvtsd2si(__a);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtsd_ss(__m128 __a, __m128d __b)
{
  __a[0] = __b[0];
  return __a;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cvtsi32_sd(__m128d __a, int __b)
{
  __a[0] = __b;
  return __a;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cvtss_sd(__m128d __a, __m128 __b)
{
  __a[0] = __b[0];
  return __a;
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvttpd_epi32(__m128d __a)
{
  return (__m128i)__builtin_ia32_cvttpd2dq(__a);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_cvttsd_si32(__m128d __a)
{
  return __a[0];
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_cvtpd_pi32(__m128d __a)
{
  return (__m64)__builtin_ia32_cvtpd2pi(__a);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_cvttpd_pi32(__m128d __a)
{
  return (__m64)__builtin_ia32_cvttpd2pi(__a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cvtpi32_pd(__m64 __a)
{
  return __builtin_ia32_cvtpi2pd((__v2si)__a);
}

static __inline__ double __DEFAULT_FN_ATTRS
_mm_cvtsd_f64(__m128d __a)
{
  return __a[0];
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_load_pd(double const *__dp)
{
  return *(__m128d*)__dp;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_load1_pd(double const *__dp)
{
  struct __mm_load1_pd_struct {
    double __u;
  } __attribute__((__packed__, __may_alias__));
  double __u = ((struct __mm_load1_pd_struct*)__dp)->__u;
  return (__m128d){ __u, __u };
}

#define        _mm_load_pd1(dp)        _mm_load1_pd(dp)

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_loadr_pd(double const *__dp)
{
  __m128d __u = *(__m128d*)__dp;
  return __builtin_shufflevector(__u, __u, 1, 0);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_loadu_pd(double const *__dp)
{
  struct __loadu_pd {
    __m128d __v;
  } __attribute__((__packed__, __may_alias__));
  return ((struct __loadu_pd*)__dp)->__v;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_load_sd(double const *__dp)
{
  struct __mm_load_sd_struct {
    double __u;
  } __attribute__((__packed__, __may_alias__));
  double __u = ((struct __mm_load_sd_struct*)__dp)->__u;
  return (__m128d){ __u, 0 };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_loadh_pd(__m128d __a, double const *__dp)
{
  struct __mm_loadh_pd_struct {
    double __u;
  } __attribute__((__packed__, __may_alias__));
  double __u = ((struct __mm_loadh_pd_struct*)__dp)->__u;
  return (__m128d){ __a[0], __u };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_loadl_pd(__m128d __a, double const *__dp)
{
  struct __mm_loadl_pd_struct {
    double __u;
  } __attribute__((__packed__, __may_alias__));
  double __u = ((struct __mm_loadl_pd_struct*)__dp)->__u;
  return (__m128d){ __u, __a[1] };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_undefined_pd()
{
  return (__m128d)__builtin_ia32_undef128();
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_set_sd(double __w)
{
  return (__m128d){ __w, 0 };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_set1_pd(double __w)
{
  return (__m128d){ __w, __w };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_set_pd(double __w, double __x)
{
  return (__m128d){ __x, __w };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_setr_pd(double __w, double __x)
{
  return (__m128d){ __w, __x };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_setzero_pd(void)
{
  return (__m128d){ 0, 0 };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_move_sd(__m128d __a, __m128d __b)
{
  return (__m128d){ __b[0], __a[1] };
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_store_sd(double *__dp, __m128d __a)
{
  struct __mm_store_sd_struct {
    double __u;
  } __attribute__((__packed__, __may_alias__));
  ((struct __mm_store_sd_struct*)__dp)->__u = __a[0];
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_store1_pd(double *__dp, __m128d __a)
{
  struct __mm_store1_pd_struct {
    double __u[2];
  } __attribute__((__packed__, __may_alias__));
  ((struct __mm_store1_pd_struct*)__dp)->__u[0] = __a[0];
  ((struct __mm_store1_pd_struct*)__dp)->__u[1] = __a[0];
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_store_pd(double *__dp, __m128d __a)
{
  *(__m128d *)__dp = __a;
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storeu_pd(double *__dp, __m128d __a)
{
  __builtin_ia32_storeupd(__dp, __a);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storer_pd(double *__dp, __m128d __a)
{
  __a = __builtin_shufflevector(__a, __a, 1, 0);
  *(__m128d *)__dp = __a;
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storeh_pd(double *__dp, __m128d __a)
{
  struct __mm_storeh_pd_struct {
    double __u;
  } __attribute__((__packed__, __may_alias__));
  ((struct __mm_storeh_pd_struct*)__dp)->__u = __a[1];
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storel_pd(double *__dp, __m128d __a)
{
  struct __mm_storeh_pd_struct {
    double __u;
  } __attribute__((__packed__, __may_alias__));
  ((struct __mm_storeh_pd_struct*)__dp)->__u = __a[0];
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_add_epi8(__m128i __a, __m128i __b)
{
  return (__m128i)((__v16qi)__a + (__v16qi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_add_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)((__v8hi)__a + (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_add_epi32(__m128i __a, __m128i __b)
{
  return (__m128i)((__v4si)__a + (__v4si)__b);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_add_si64(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_paddq(__a, __b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_add_epi64(__m128i __a, __m128i __b)
{
  return __a + __b;
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_adds_epi8(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_paddsb128((__v16qi)__a, (__v16qi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_adds_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_paddsw128((__v8hi)__a, (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_adds_epu8(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_paddusb128((__v16qi)__a, (__v16qi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_adds_epu16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_paddusw128((__v8hi)__a, (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_avg_epu8(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_pavgb128((__v16qi)__a, (__v16qi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_avg_epu16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_pavgw128((__v8hi)__a, (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_madd_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_pmaddwd128((__v8hi)__a, (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_max_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_pmaxsw128((__v8hi)__a, (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_max_epu8(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_pmaxub128((__v16qi)__a, (__v16qi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_min_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_pminsw128((__v8hi)__a, (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_min_epu8(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_pminub128((__v16qi)__a, (__v16qi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mulhi_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_pmulhw128((__v8hi)__a, (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mulhi_epu16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_pmulhuw128((__v8hi)__a, (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mullo_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)((__v8hi)__a * (__v8hi)__b);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_mul_su32(__m64 __a, __m64 __b)
{
  return __builtin_ia32_pmuludq((__v2si)__a, (__v2si)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mul_epu32(__m128i __a, __m128i __b)
{
  return __builtin_ia32_pmuludq128((__v4si)__a, (__v4si)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sad_epu8(__m128i __a, __m128i __b)
{
  return __builtin_ia32_psadbw128((__v16qi)__a, (__v16qi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sub_epi8(__m128i __a, __m128i __b)
{
  return (__m128i)((__v16qi)__a - (__v16qi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sub_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)((__v8hi)__a - (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sub_epi32(__m128i __a, __m128i __b)
{
  return (__m128i)((__v4si)__a - (__v4si)__b);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_sub_si64(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_psubq(__a, __b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sub_epi64(__m128i __a, __m128i __b)
{
  return __a - __b;
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_subs_epi8(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_psubsb128((__v16qi)__a, (__v16qi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_subs_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_psubsw128((__v8hi)__a, (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_subs_epu8(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_psubusb128((__v16qi)__a, (__v16qi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_subs_epu16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_psubusw128((__v8hi)__a, (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_and_si128(__m128i __a, __m128i __b)
{
  return __a & __b;
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_andnot_si128(__m128i __a, __m128i __b)
{
  return ~__a & __b;
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_or_si128(__m128i __a, __m128i __b)
{
  return __a | __b;
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_xor_si128(__m128i __a, __m128i __b)
{
  return __a ^ __b;
}

#define _mm_slli_si128(a, imm) __extension__ ({                         \
  (__m128i)__builtin_shufflevector((__v16qi)_mm_setzero_si128(),        \
                                   (__v16qi)(__m128i)(a),               \
                                   ((imm)&0xF0) ? 0 : 16 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 17 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 18 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 19 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 20 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 21 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 22 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 23 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 24 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 25 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 26 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 27 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 28 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 29 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 30 - ((imm)&0xF), \
                                   ((imm)&0xF0) ? 0 : 31 - ((imm)&0xF)); })

#define _mm_bslli_si128(a, imm) \
  _mm_slli_si128((a), (imm))

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_slli_epi16(__m128i __a, int __count)
{
  return (__m128i)__builtin_ia32_psllwi128((__v8hi)__a, __count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sll_epi16(__m128i __a, __m128i __count)
{
  return (__m128i)__builtin_ia32_psllw128((__v8hi)__a, (__v8hi)__count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_slli_epi32(__m128i __a, int __count)
{
  return (__m128i)__builtin_ia32_pslldi128((__v4si)__a, __count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sll_epi32(__m128i __a, __m128i __count)
{
  return (__m128i)__builtin_ia32_pslld128((__v4si)__a, (__v4si)__count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_slli_epi64(__m128i __a, int __count)
{
  return __builtin_ia32_psllqi128(__a, __count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sll_epi64(__m128i __a, __m128i __count)
{
  return __builtin_ia32_psllq128(__a, __count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srai_epi16(__m128i __a, int __count)
{
  return (__m128i)__builtin_ia32_psrawi128((__v8hi)__a, __count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sra_epi16(__m128i __a, __m128i __count)
{
  return (__m128i)__builtin_ia32_psraw128((__v8hi)__a, (__v8hi)__count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srai_epi32(__m128i __a, int __count)
{
  return (__m128i)__builtin_ia32_psradi128((__v4si)__a, __count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sra_epi32(__m128i __a, __m128i __count)
{
  return (__m128i)__builtin_ia32_psrad128((__v4si)__a, (__v4si)__count);
}

#define _mm_srli_si128(a, imm) __extension__ ({                          \
  (__m128i)__builtin_shufflevector((__v16qi)(__m128i)(a),                \
                                   (__v16qi)_mm_setzero_si128(),         \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 0,  \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 1,  \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 2,  \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 3,  \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 4,  \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 5,  \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 6,  \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 7,  \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 8,  \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 9,  \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 10, \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 11, \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 12, \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 13, \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 14, \
                                   ((imm)&0xF0) ? 16 : ((imm)&0xF) + 15); })

#define _mm_bsrli_si128(a, imm) \
  _mm_srli_si128((a), (imm))

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srli_epi16(__m128i __a, int __count)
{
  return (__m128i)__builtin_ia32_psrlwi128((__v8hi)__a, __count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srl_epi16(__m128i __a, __m128i __count)
{
  return (__m128i)__builtin_ia32_psrlw128((__v8hi)__a, (__v8hi)__count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srli_epi32(__m128i __a, int __count)
{
  return (__m128i)__builtin_ia32_psrldi128((__v4si)__a, __count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srl_epi32(__m128i __a, __m128i __count)
{
  return (__m128i)__builtin_ia32_psrld128((__v4si)__a, (__v4si)__count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srli_epi64(__m128i __a, int __count)
{
  return __builtin_ia32_psrlqi128(__a, __count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srl_epi64(__m128i __a, __m128i __count)
{
  return __builtin_ia32_psrlq128(__a, __count);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpeq_epi8(__m128i __a, __m128i __b)
{
  return (__m128i)((__v16qi)__a == (__v16qi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpeq_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)((__v8hi)__a == (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpeq_epi32(__m128i __a, __m128i __b)
{
  return (__m128i)((__v4si)__a == (__v4si)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpgt_epi8(__m128i __a, __m128i __b)
{
  /* This function always performs a signed comparison, but __v16qi is a char
     which may be signed or unsigned, so use __v16qs. */
  return (__m128i)((__v16qs)__a > (__v16qs)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpgt_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)((__v8hi)__a > (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpgt_epi32(__m128i __a, __m128i __b)
{
  return (__m128i)((__v4si)__a > (__v4si)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmplt_epi8(__m128i __a, __m128i __b)
{
  return _mm_cmpgt_epi8(__b, __a);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmplt_epi16(__m128i __a, __m128i __b)
{
  return _mm_cmpgt_epi16(__b, __a);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmplt_epi32(__m128i __a, __m128i __b)
{
  return _mm_cmpgt_epi32(__b, __a);
}

#ifdef __x86_64__
static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cvtsi64_sd(__m128d __a, long long __b)
{
  __a[0] = __b;
  return __a;
}

static __inline__ long long __DEFAULT_FN_ATTRS
_mm_cvtsd_si64(__m128d __a)
{
  return __builtin_ia32_cvtsd2si64(__a);
}

static __inline__ long long __DEFAULT_FN_ATTRS
_mm_cvttsd_si64(__m128d __a)
{
  return __a[0];
}
#endif

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtepi32_ps(__m128i __a)
{
  return __builtin_ia32_cvtdq2ps((__v4si)__a);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvtps_epi32(__m128 __a)
{
  return (__m128i)__builtin_ia32_cvtps2dq(__a);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvttps_epi32(__m128 __a)
{
  return (__m128i)__builtin_ia32_cvttps2dq(__a);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvtsi32_si128(int __a)
{
  return (__m128i)(__v4si){ __a, 0, 0, 0 };
}

#ifdef __x86_64__
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvtsi64_si128(long long __a)
{
  return (__m128i){ __a, 0 };
}
#endif

static __inline__ int __DEFAULT_FN_ATTRS
_mm_cvtsi128_si32(__m128i __a)
{
  __v4si __b = (__v4si)__a;
  return __b[0];
}

#ifdef __x86_64__
static __inline__ long long __DEFAULT_FN_ATTRS
_mm_cvtsi128_si64(__m128i __a)
{
  return __a[0];
}
#endif

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_load_si128(__m128i const *__p)
{
  return *__p;
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_loadu_si128(__m128i const *__p)
{
  struct __loadu_si128 {
    __m128i __v;
  } __attribute__((__packed__, __may_alias__));
  return ((struct __loadu_si128*)__p)->__v;
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_loadl_epi64(__m128i const *__p)
{
  struct __mm_loadl_epi64_struct {
    long long __u;
  } __attribute__((__packed__, __may_alias__));
  return (__m128i) { ((struct __mm_loadl_epi64_struct*)__p)->__u, 0};
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_undefined_si128()
{
  return (__m128i)__builtin_ia32_undef128();
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set_epi64x(long long __q1, long long __q0)
{
  return (__m128i){ __q0, __q1 };
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set_epi64(__m64 __q1, __m64 __q0)
{
  return (__m128i){ (long long)__q0, (long long)__q1 };
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set_epi32(int __i3, int __i2, int __i1, int __i0)
{
  return (__m128i)(__v4si){ __i0, __i1, __i2, __i3};
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set_epi16(short __w7, short __w6, short __w5, short __w4, short __w3, short __w2, short __w1, short __w0)
{
  return (__m128i)(__v8hi){ __w0, __w1, __w2, __w3, __w4, __w5, __w6, __w7 };
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set_epi8(char __b15, char __b14, char __b13, char __b12, char __b11, char __b10, char __b9, char __b8, char __b7, char __b6, char __b5, char __b4, char __b3, char __b2, char __b1, char __b0)
{
  return (__m128i)(__v16qi){ __b0, __b1, __b2, __b3, __b4, __b5, __b6, __b7, __b8, __b9, __b10, __b11, __b12, __b13, __b14, __b15 };
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set1_epi64x(long long __q)
{
  return (__m128i){ __q, __q };
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set1_epi64(__m64 __q)
{
  return (__m128i){ (long long)__q, (long long)__q };
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set1_epi32(int __i)
{
  return (__m128i)(__v4si){ __i, __i, __i, __i };
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set1_epi16(short __w)
{
  return (__m128i)(__v8hi){ __w, __w, __w, __w, __w, __w, __w, __w };
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set1_epi8(char __b)
{
  return (__m128i)(__v16qi){ __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b };
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_setr_epi64(__m64 __q0, __m64 __q1)
{
  return (__m128i){ (long long)__q0, (long long)__q1 };
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_setr_epi32(int __i0, int __i1, int __i2, int __i3)
{
  return (__m128i)(__v4si){ __i0, __i1, __i2, __i3};
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_setr_epi16(short __w0, short __w1, short __w2, short __w3, short __w4, short __w5, short __w6, short __w7)
{
  return (__m128i)(__v8hi){ __w0, __w1, __w2, __w3, __w4, __w5, __w6, __w7 };
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_setr_epi8(char __b0, char __b1, char __b2, char __b3, char __b4, char __b5, char __b6, char __b7, char __b8, char __b9, char __b10, char __b11, char __b12, char __b13, char __b14, char __b15)
{
  return (__m128i)(__v16qi){ __b0, __b1, __b2, __b3, __b4, __b5, __b6, __b7, __b8, __b9, __b10, __b11, __b12, __b13, __b14, __b15 };
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_setzero_si128(void)
{
  return (__m128i){ 0LL, 0LL };
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_store_si128(__m128i *__p, __m128i __b)
{
  *__p = __b;
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storeu_si128(__m128i *__p, __m128i __b)
{
  __builtin_ia32_storedqu((char *)__p, (__v16qi)__b);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_maskmoveu_si128(__m128i __d, __m128i __n, char *__p)
{
  __builtin_ia32_maskmovdqu((__v16qi)__d, (__v16qi)__n, __p);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storel_epi64(__m128i *__p, __m128i __a)
{
  struct __mm_storel_epi64_struct {
    long long __u;
  } __attribute__((__packed__, __may_alias__));
  ((struct __mm_storel_epi64_struct*)__p)->__u = __a[0];
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_stream_pd(double *__p, __m128d __a)
{
  __builtin_ia32_movntpd(__p, __a);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_stream_si128(__m128i *__p, __m128i __a)
{
  __builtin_ia32_movntdq(__p, __a);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_stream_si32(int *__p, int __a)
{
  __builtin_ia32_movnti(__p, __a);
}

#ifdef __x86_64__
static __inline__ void __DEFAULT_FN_ATTRS
_mm_stream_si64(long long *__p, long long __a)
{
  __builtin_ia32_movnti64(__p, __a);
}
#endif

static __inline__ void __DEFAULT_FN_ATTRS
_mm_clflush(void const *__p)
{
  __builtin_ia32_clflush(__p);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_lfence(void)
{
  __builtin_ia32_lfence();
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_mfence(void)
{
  __builtin_ia32_mfence();
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_packs_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_packsswb128((__v8hi)__a, (__v8hi)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_packs_epi32(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_packssdw128((__v4si)__a, (__v4si)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_packus_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_packuswb128((__v8hi)__a, (__v8hi)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_extract_epi16(__m128i __a, int __imm)
{
  __v8hi __b = (__v8hi)__a;
  return (unsigned short)__b[__imm & 7];
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_insert_epi16(__m128i __a, int __b, int __imm)
{
  __v8hi __c = (__v8hi)__a;
  __c[__imm & 7] = __b;
  return (__m128i)__c;
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_movemask_epi8(__m128i __a)
{
  return __builtin_ia32_pmovmskb128((__v16qi)__a);
}

#define _mm_shuffle_epi32(a, imm) __extension__ ({ \
  (__m128i)__builtin_shufflevector((__v4si)(__m128i)(a), \
                                   (__v4si)_mm_setzero_si128(), \
                                   (imm) & 0x3, ((imm) & 0xc) >> 2, \
                                   ((imm) & 0x30) >> 4, ((imm) & 0xc0) >> 6); })

#define _mm_shufflelo_epi16(a, imm) __extension__ ({ \
  (__m128i)__builtin_shufflevector((__v8hi)(__m128i)(a), \
                                   (__v8hi)_mm_setzero_si128(), \
                                   (imm) & 0x3, ((imm) & 0xc) >> 2, \
                                   ((imm) & 0x30) >> 4, ((imm) & 0xc0) >> 6, \
                                   4, 5, 6, 7); })

#define _mm_shufflehi_epi16(a, imm) __extension__ ({ \
  (__m128i)__builtin_shufflevector((__v8hi)(__m128i)(a), \
                                   (__v8hi)_mm_setzero_si128(), \
                                   0, 1, 2, 3, \
                                   4 + (((imm) & 0x03) >> 0), \
                                   4 + (((imm) & 0x0c) >> 2), \
                                   4 + (((imm) & 0x30) >> 4), \
                                   4 + (((imm) & 0xc0) >> 6)); })

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_unpackhi_epi8(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_shufflevector((__v16qi)__a, (__v16qi)__b, 8, 16+8, 9, 16+9, 10, 16+10, 11, 16+11, 12, 16+12, 13, 16+13, 14, 16+14, 15, 16+15);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_unpackhi_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_shufflevector((__v8hi)__a, (__v8hi)__b, 4, 8+4, 5, 8+5, 6, 8+6, 7, 8+7);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_unpackhi_epi32(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_shufflevector((__v4si)__a, (__v4si)__b, 2, 4+2, 3, 4+3);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_unpackhi_epi64(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_shufflevector(__a, __b, 1, 2+1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_unpacklo_epi8(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_shufflevector((__v16qi)__a, (__v16qi)__b, 0, 16+0, 1, 16+1, 2, 16+2, 3, 16+3, 4, 16+4, 5, 16+5, 6, 16+6, 7, 16+7);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_unpacklo_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_shufflevector((__v8hi)__a, (__v8hi)__b, 0, 8+0, 1, 8+1, 2, 8+2, 3, 8+3);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_unpacklo_epi32(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_shufflevector((__v4si)__a, (__v4si)__b, 0, 4+0, 1, 4+1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_unpacklo_epi64(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_shufflevector(__a, __b, 0, 2+0);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_movepi64_pi64(__m128i __a)
{
  return (__m64)__a[0];
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_movpi64_epi64(__m64 __a)
{
  return (__m128i){ (long long)__a, 0 };
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_move_epi64(__m128i __a)
{
  return __builtin_shufflevector(__a, (__m128i){ 0 }, 0, 2);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_unpackhi_pd(__m128d __a, __m128d __b)
{
  return __builtin_shufflevector(__a, __b, 1, 2+1);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_unpacklo_pd(__m128d __a, __m128d __b)
{
  return __builtin_shufflevector(__a, __b, 0, 2+0);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_movemask_pd(__m128d __a)
{
  return __builtin_ia32_movmskpd(__a);
}

#define _mm_shuffle_pd(a, b, i) __extension__ ({ \
  (__m128d)__builtin_shufflevector((__v2df)(__m128d)(a), (__v2df)(__m128d)(b), \
                                   (i) & 1, (((i) & 2) >> 1) + 2); })

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_castpd_ps(__m128d __a)
{
  return (__m128)__a;
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_castpd_si128(__m128d __a)
{
  return (__m128i)__a;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_castps_pd(__m128 __a)
{
  return (__m128d)__a;
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_castps_si128(__m128 __a)
{
  return (__m128i)__a;
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_castsi128_ps(__m128i __a)
{
  return (__m128)__a;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_castsi128_pd(__m128i __a)
{
  return (__m128d)__a;
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_pause(void)
{
  __builtin_ia32_pause();
}

#undef __DEFAULT_FN_ATTRS

#define _MM_SHUFFLE2(x, y) (((x) << 1) | (y))

#endif /* __EMMINTRIN_H */
