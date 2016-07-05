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

/* Unsigned types */
typedef unsigned long long __v2du __attribute__ ((__vector_size__ (16)));
typedef unsigned short __v8hu __attribute__((__vector_size__(16)));
typedef unsigned char __v16qu __attribute__((__vector_size__(16)));

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
  return (__m128d)((__v2df)__a + (__v2df)__b);
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
  return (__m128d)((__v2df)__a - (__v2df)__b);
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
  return (__m128d)((__v2df)__a * (__v2df)__b);
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
  return (__m128d)((__v2df)__a / (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_sqrt_sd(__m128d __a, __m128d __b)
{
  __m128d __c = __builtin_ia32_sqrtsd((__v2df)__b);
  return (__m128d) { __c[0], __a[1] };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_sqrt_pd(__m128d __a)
{
  return __builtin_ia32_sqrtpd((__v2df)__a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_min_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_minsd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_min_pd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_minpd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_max_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_maxsd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_max_pd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_maxpd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_and_pd(__m128d __a, __m128d __b)
{
  return (__m128d)((__v4su)__a & (__v4su)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_andnot_pd(__m128d __a, __m128d __b)
{
  return (__m128d)(~(__v4su)__a & (__v4su)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_or_pd(__m128d __a, __m128d __b)
{
  return (__m128d)((__v4su)__a | (__v4su)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_xor_pd(__m128d __a, __m128d __b)
{
  return (__m128d)((__v4su)__a ^ (__v4su)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpeq_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpeqpd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmplt_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpltpd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmple_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmplepd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpgt_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpltpd((__v2df)__b, (__v2df)__a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpge_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmplepd((__v2df)__b, (__v2df)__a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpord_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpordpd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpunord_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpunordpd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpneq_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpneqpd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpnlt_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpnltpd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpnle_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpnlepd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpngt_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpnltpd((__v2df)__b, (__v2df)__a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpnge_pd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpnlepd((__v2df)__b, (__v2df)__a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpeq_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpeqsd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmplt_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpltsd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmple_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmplesd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpgt_sd(__m128d __a, __m128d __b)
{
  __m128d __c = __builtin_ia32_cmpltsd((__v2df)__b, (__v2df)__a);
  return (__m128d) { __c[0], __a[1] };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpge_sd(__m128d __a, __m128d __b)
{
  __m128d __c = __builtin_ia32_cmplesd((__v2df)__b, (__v2df)__a);
  return (__m128d) { __c[0], __a[1] };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpord_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpordsd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpunord_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpunordsd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpneq_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpneqsd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpnlt_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpnltsd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpnle_sd(__m128d __a, __m128d __b)
{
  return (__m128d)__builtin_ia32_cmpnlesd((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpngt_sd(__m128d __a, __m128d __b)
{
  __m128d __c = __builtin_ia32_cmpnltsd((__v2df)__b, (__v2df)__a);
  return (__m128d) { __c[0], __a[1] };
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cmpnge_sd(__m128d __a, __m128d __b)
{
  __m128d __c = __builtin_ia32_cmpnlesd((__v2df)__b, (__v2df)__a);
  return (__m128d) { __c[0], __a[1] };
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_comieq_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_comisdeq((__v2df)__a, (__v2df)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_comilt_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_comisdlt((__v2df)__a, (__v2df)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_comile_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_comisdle((__v2df)__a, (__v2df)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_comigt_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_comisdgt((__v2df)__a, (__v2df)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_comige_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_comisdge((__v2df)__a, (__v2df)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_comineq_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_comisdneq((__v2df)__a, (__v2df)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomieq_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_ucomisdeq((__v2df)__a, (__v2df)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomilt_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_ucomisdlt((__v2df)__a, (__v2df)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomile_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_ucomisdle((__v2df)__a, (__v2df)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomigt_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_ucomisdgt((__v2df)__a, (__v2df)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomige_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_ucomisdge((__v2df)__a, (__v2df)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomineq_sd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_ucomisdneq((__v2df)__a, (__v2df)__b);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtpd_ps(__m128d __a)
{
  return __builtin_ia32_cvtpd2ps((__v2df)__a);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cvtps_pd(__m128 __a)
{
  return (__m128d) __builtin_convertvector(
      __builtin_shufflevector((__v4sf)__a, (__v4sf)__a, 0, 1), __v2df);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cvtepi32_pd(__m128i __a)
{
  return (__m128d) __builtin_convertvector(
      __builtin_shufflevector((__v4si)__a, (__v4si)__a, 0, 1), __v2df);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvtpd_epi32(__m128d __a)
{
  return __builtin_ia32_cvtpd2dq((__v2df)__a);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_cvtsd_si32(__m128d __a)
{
  return __builtin_ia32_cvtsd2si((__v2df)__a);
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
  return (__m128i)__builtin_ia32_cvttpd2dq((__v2df)__a);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_cvttsd_si32(__m128d __a)
{
  return __a[0];
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_cvtpd_pi32(__m128d __a)
{
  return (__m64)__builtin_ia32_cvtpd2pi((__v2df)__a);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_cvttpd_pi32(__m128d __a)
{
  return (__m64)__builtin_ia32_cvttpd2pi((__v2df)__a);
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
  return __builtin_shufflevector((__v2df)__u, (__v2df)__u, 1, 0);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_loadu_pd(double const *__dp)
{
  struct __loadu_pd {
    __m128d __v;
  } __attribute__((__packed__, __may_alias__));
  return ((struct __loadu_pd*)__dp)->__v;
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_loadu_si64(void const *__a)
{
  struct __loadu_si64 {
    long long __v;
  } __attribute__((__packed__, __may_alias__));
  long long __u = ((struct __loadu_si64*)__a)->__v;
  return (__m128i){__u, 0L};
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
_mm_undefined_pd(void)
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
_mm_store_pd(double *__dp, __m128d __a)
{
  *(__m128d*)__dp = __a;
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_store1_pd(double *__dp, __m128d __a)
{
  __a = __builtin_shufflevector((__v2df)__a, (__v2df)__a, 0, 0);
  _mm_store_pd(__dp, __a);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_store_pd1(double *__dp, __m128d __a)
{
  return _mm_store1_pd(__dp, __a);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storeu_pd(double *__dp, __m128d __a)
{
  struct __storeu_pd {
    __m128d __v;
  } __attribute__((__packed__, __may_alias__));
  ((struct __storeu_pd*)__dp)->__v = __a;
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storer_pd(double *__dp, __m128d __a)
{
  __a = __builtin_shufflevector((__v2df)__a, (__v2df)__a, 1, 0);
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
  return (__m128i)((__v16qu)__a + (__v16qu)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_add_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)((__v8hu)__a + (__v8hu)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_add_epi32(__m128i __a, __m128i __b)
{
  return (__m128i)((__v4su)__a + (__v4su)__b);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_add_si64(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_paddq((__v1di)__a, (__v1di)__b);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_add_epi64(__m128i __a, __m128i __b)
{
  return (__m128i)((__v2du)__a + (__v2du)__b);
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

/// \brief Multiplies the corresponding elements of two [8 x short] vectors and
///    returns a vector containing the low-order 16 bits of each 32-bit product
///    in the corresponding element.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPMULLW / PMULLW instruction.
///
/// \param __a
///    A 128-bit integer vector containing one of the source operands.
/// \param __b
///    A 128-bit integer vector containing one of the source operands.
/// \returns A 128-bit integer vector containing the products of both operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mullo_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)((__v8hu)__a * (__v8hu)__b);
}

/// \brief Multiplies 32-bit unsigned integer values contained in the lower bits
///    of the two 64-bit integer vectors and returns the 64-bit unsigned
///    product.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c PMULUDQ instruction.
///
/// \param __a
///    A 64-bit integer containing one of the source operands.
/// \param __b
///    A 64-bit integer containing one of the source operands.
/// \returns A 64-bit integer vector containing the product of both operands.
static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_mul_su32(__m64 __a, __m64 __b)
{
  return __builtin_ia32_pmuludq((__v2si)__a, (__v2si)__b);
}

/// \brief Multiplies 32-bit unsigned integer values contained in the lower
///    bits of the corresponding elements of two [2 x i64] vectors, and returns
///    the 64-bit products in the corresponding elements of a [2 x i64] vector.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPMULUDQ / PMULUDQ instruction.
///
/// \param __a
///    A [2 x i64] vector containing one of the source operands.
/// \param __b
///    A [2 x i64] vector containing one of the source operands.
/// \returns A [2 x i64] vector containing the product of both operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_mul_epu32(__m128i __a, __m128i __b)
{
  return __builtin_ia32_pmuludq128((__v4si)__a, (__v4si)__b);
}

/// \brief Computes the absolute differences of corresponding 8-bit integer
///    values in two 128-bit vectors. Sums the first 8 absolute differences, and
///    separately sums the second 8 absolute differences. Packss these two
///    unsigned 16-bit integer sums into the upper and lower elements of a
///    [2 x i64] vector.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSADBW / PSADBW instruction.
///
/// \param __a
///    A 128-bit integer vector containing one of the source operands.
/// \param __b
///    A 128-bit integer vector containing one of the source operands.
/// \returns A [2 x i64] vector containing the sums of the sets of absolute
///    differences between both operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sad_epu8(__m128i __a, __m128i __b)
{
  return __builtin_ia32_psadbw128((__v16qi)__a, (__v16qi)__b);
}

/// \brief Subtracts the corresponding 8-bit integer values in the operands.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSUBB / PSUBB instruction.
///
/// \param __a
///    A 128-bit integer vector containing the minuends.
/// \param __b
///    A 128-bit integer vector containing the subtrahends.
/// \returns A 128-bit integer vector containing the differences of the values
///    in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sub_epi8(__m128i __a, __m128i __b)
{
  return (__m128i)((__v16qu)__a - (__v16qu)__b);
}

/// \brief Subtracts the corresponding 16-bit integer values in the operands.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSUBW / PSUBW instruction.
///
/// \param __a
///    A 128-bit integer vector containing the minuends.
/// \param __b
///    A 128-bit integer vector containing the subtrahends.
/// \returns A 128-bit integer vector containing the differences of the values
///    in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sub_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)((__v8hu)__a - (__v8hu)__b);
}

/// \brief Subtracts the corresponding 32-bit integer values in the operands.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSUBD / PSUBD instruction.
///
/// \param __a
///    A 128-bit integer vector containing the minuends.
/// \param __b
///    A 128-bit integer vector containing the subtrahends.
/// \returns A 128-bit integer vector containing the differences of the values
///    in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sub_epi32(__m128i __a, __m128i __b)
{
  return (__m128i)((__v4su)__a - (__v4su)__b);
}

/// \brief Subtracts signed or unsigned 64-bit integer values and writes the
///    difference to the corresponding bits in the destination.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c PSUBQ instruction.
///
/// \param __a
///    A 64-bit integer vector containing the minuend.
/// \param __b
///    A 64-bit integer vector containing the subtrahend.
/// \returns A 64-bit integer vector containing the difference of the values in
///    the operands.
static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_sub_si64(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_psubq((__v1di)__a, (__v1di)__b);
}

/// \brief Subtracts the corresponding elements of two [2 x i64] vectors.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSUBQ / PSUBQ instruction.
///
/// \param __a
///    A 128-bit integer vector containing the minuends.
/// \param __b
///    A 128-bit integer vector containing the subtrahends.
/// \returns A 128-bit integer vector containing the differences of the values
///    in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sub_epi64(__m128i __a, __m128i __b)
{
  return (__m128i)((__v2du)__a - (__v2du)__b);
}

/// \brief Subtracts corresponding 8-bit signed integer values in the input and
///    returns the differences in the corresponding bytes in the destination.
///    Differences greater than 7Fh are saturated to 7Fh, and differences less
///    than 80h are saturated to 80h.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSUBSB / PSUBSB instruction.
///
/// \param __a
///    A 128-bit integer vector containing the minuends.
/// \param __b
///    A 128-bit integer vector containing the subtrahends.
/// \returns A 128-bit integer vector containing the differences of the values
///    in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_subs_epi8(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_psubsb128((__v16qi)__a, (__v16qi)__b);
}

/// \brief Subtracts corresponding 16-bit signed integer values in the input and
///    returns the differences in the corresponding bytes in the destination.
///    Differences greater than 7FFFh are saturated to 7FFFh, and values less
///    than 8000h are saturated to 8000h.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSUBSW / PSUBSW instruction.
///
/// \param __a
///    A 128-bit integer vector containing the minuends.
/// \param __b
///    A 128-bit integer vector containing the subtrahends.
/// \returns A 128-bit integer vector containing the differences of the values
///    in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_subs_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_psubsw128((__v8hi)__a, (__v8hi)__b);
}

/// \brief Subtracts corresponding 8-bit unsigned integer values in the input
///    and returns the differences in the corresponding bytes in the
///    destination. Differences less than 00h are saturated to 00h.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSUBUSB / PSUBUSB instruction.
///
/// \param __a
///    A 128-bit integer vector containing the minuends.
/// \param __b
///    A 128-bit integer vector containing the subtrahends.
/// \returns A 128-bit integer vector containing the unsigned integer
///    differences of the values in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_subs_epu8(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_psubusb128((__v16qi)__a, (__v16qi)__b);
}

/// \brief Subtracts corresponding 16-bit unsigned integer values in the input
///    and returns the differences in the corresponding bytes in the
///    destination. Differences less than 0000h are saturated to 0000h.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSUBUSW / PSUBUSW instruction.
///
/// \param __a
///    A 128-bit integer vector containing the minuends.
/// \param __b
///    A 128-bit integer vector containing the subtrahends.
/// \returns A 128-bit integer vector containing the unsigned integer
///    differences of the values in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_subs_epu16(__m128i __a, __m128i __b)
{
  return (__m128i)__builtin_ia32_psubusw128((__v8hi)__a, (__v8hi)__b);
}

/// \brief Performs a bitwise AND of two 128-bit integer vectors.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPAND / PAND instruction.
///
/// \param __a
///    A 128-bit integer vector containing one of the source operands.
/// \param __b
///    A 128-bit integer vector containing one of the source operands.
/// \returns A 128-bit integer vector containing the bitwise AND of the values
///    in both operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_and_si128(__m128i __a, __m128i __b)
{
  return (__m128i)((__v2du)__a & (__v2du)__b);
}

/// \brief Performs a bitwise AND of two 128-bit integer vectors, using the
///    one's complement of the values contained in the first source operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPANDN / PANDN instruction.
///
/// \param __a
///    A 128-bit vector containing the left source operand. The one's complement
///    of this value is used in the bitwise AND.
/// \param __b
///    A 128-bit vector containing the right source operand.
/// \returns A 128-bit integer vector containing the bitwise AND of the one's
///    complement of the first operand and the values in the second operand.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_andnot_si128(__m128i __a, __m128i __b)
{
  return (__m128i)(~(__v2du)__a & (__v2du)__b);
}
/// \brief Performs a bitwise OR of two 128-bit integer vectors.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPOR / POR instruction.
///
/// \param __a
///    A 128-bit integer vector containing one of the source operands.
/// \param __b
///    A 128-bit integer vector containing one of the source operands.
/// \returns A 128-bit integer vector containing the bitwise OR of the values
///    in both operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_or_si128(__m128i __a, __m128i __b)
{
  return (__m128i)((__v2du)__a | (__v2du)__b);
}

/// \brief Performs a bitwise exclusive OR of two 128-bit integer vectors.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPXOR / PXOR instruction.
///
/// \param __a
///    A 128-bit integer vector containing one of the source operands.
/// \param __b
///    A 128-bit integer vector containing one of the source operands.
/// \returns A 128-bit integer vector containing the bitwise exclusive OR of the
///    values in both operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_xor_si128(__m128i __a, __m128i __b)
{
  return (__m128i)((__v2du)__a ^ (__v2du)__b);
}

/// \brief Left-shifts the 128-bit integer vector operand by the specified
///    number of bytes. Low-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// __m128i _mm_slli_si128(__m128i a, const int imm);
/// \endcode
///
/// This intrinsic corresponds to the \c VPSLLDQ / PSLLDQ instruction.
///
/// \param a
///    A 128-bit integer vector containing the source operand.
/// \param imm
///    An immediate value specifying the number of bytes to left-shift
///    operand a.
/// \returns A 128-bit integer vector containing the left-shifted value.
#define _mm_slli_si128(a, imm) __extension__ ({                              \
  (__m128i)__builtin_shufflevector(                                          \
                                 (__v16qi)_mm_setzero_si128(),               \
                                 (__v16qi)(__m128i)(a),                      \
                                 ((char)(imm)&0xF0) ?  0 : 16 - (char)(imm), \
                                 ((char)(imm)&0xF0) ?  1 : 17 - (char)(imm), \
                                 ((char)(imm)&0xF0) ?  2 : 18 - (char)(imm), \
                                 ((char)(imm)&0xF0) ?  3 : 19 - (char)(imm), \
                                 ((char)(imm)&0xF0) ?  4 : 20 - (char)(imm), \
                                 ((char)(imm)&0xF0) ?  5 : 21 - (char)(imm), \
                                 ((char)(imm)&0xF0) ?  6 : 22 - (char)(imm), \
                                 ((char)(imm)&0xF0) ?  7 : 23 - (char)(imm), \
                                 ((char)(imm)&0xF0) ?  8 : 24 - (char)(imm), \
                                 ((char)(imm)&0xF0) ?  9 : 25 - (char)(imm), \
                                 ((char)(imm)&0xF0) ? 10 : 26 - (char)(imm), \
                                 ((char)(imm)&0xF0) ? 11 : 27 - (char)(imm), \
                                 ((char)(imm)&0xF0) ? 12 : 28 - (char)(imm), \
                                 ((char)(imm)&0xF0) ? 13 : 29 - (char)(imm), \
                                 ((char)(imm)&0xF0) ? 14 : 30 - (char)(imm), \
                                 ((char)(imm)&0xF0) ? 15 : 31 - (char)(imm)); })

#define _mm_bslli_si128(a, imm) \
  _mm_slli_si128((a), (imm))

/// \brief Left-shifts each 16-bit value in the 128-bit integer vector operand
///    by the specified number of bits. Low-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSLLW / PSLLW instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    An integer value specifying the number of bits to left-shift each value
///    in operand __a.
/// \returns A 128-bit integer vector containing the left-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_slli_epi16(__m128i __a, int __count)
{
  return (__m128i)__builtin_ia32_psllwi128((__v8hi)__a, __count);
}

/// \brief Left-shifts each 16-bit value in the 128-bit integer vector operand
///    by the specified number of bits. Low-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSLLW / PSLLW instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    A 128-bit integer vector in which bits [63:0] specify the number of bits
///    to left-shift each value in operand __a.
/// \returns A 128-bit integer vector containing the left-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sll_epi16(__m128i __a, __m128i __count)
{
  return (__m128i)__builtin_ia32_psllw128((__v8hi)__a, (__v8hi)__count);
}

/// \brief Left-shifts each 32-bit value in the 128-bit integer vector operand
///    by the specified number of bits. Low-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSLLD / PSLLD instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    An integer value specifying the number of bits to left-shift each value
///    in operand __a.
/// \returns A 128-bit integer vector containing the left-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_slli_epi32(__m128i __a, int __count)
{
  return (__m128i)__builtin_ia32_pslldi128((__v4si)__a, __count);
}

/// \brief Left-shifts each 32-bit value in the 128-bit integer vector operand
///    by the specified number of bits. Low-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSLLD / PSLLD instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    A 128-bit integer vector in which bits [63:0] specify the number of bits
///    to left-shift each value in operand __a.
/// \returns A 128-bit integer vector containing the left-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sll_epi32(__m128i __a, __m128i __count)
{
  return (__m128i)__builtin_ia32_pslld128((__v4si)__a, (__v4si)__count);
}

/// \brief Left-shifts each 64-bit value in the 128-bit integer vector operand
///    by the specified number of bits. Low-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSLLQ / PSLLQ instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    An integer value specifying the number of bits to left-shift each value
///    in operand __a.
/// \returns A 128-bit integer vector containing the left-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_slli_epi64(__m128i __a, int __count)
{
  return __builtin_ia32_psllqi128((__v2di)__a, __count);
}

/// \brief Left-shifts each 64-bit value in the 128-bit integer vector operand
///    by the specified number of bits. Low-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSLLQ / PSLLQ instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    A 128-bit integer vector in which bits [63:0] specify the number of bits
///    to left-shift each value in operand __a.
/// \returns A 128-bit integer vector containing the left-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sll_epi64(__m128i __a, __m128i __count)
{
  return __builtin_ia32_psllq128((__v2di)__a, (__v2di)__count);
}

/// \brief Right-shifts each 16-bit value in the 128-bit integer vector operand
///    by the specified number of bits. High-order bits are filled with the sign
///    bit of the initial value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSRAW / PSRAW instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    An integer value specifying the number of bits to right-shift each value
///    in operand __a.
/// \returns A 128-bit integer vector containing the right-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srai_epi16(__m128i __a, int __count)
{
  return (__m128i)__builtin_ia32_psrawi128((__v8hi)__a, __count);
}

/// \brief Right-shifts each 16-bit value in the 128-bit integer vector operand
///    by the specified number of bits. High-order bits are filled with the sign
///    bit of the initial value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSRAW / PSRAW instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    A 128-bit integer vector in which bits [63:0] specify the number of bits
///    to right-shift each value in operand __a.
/// \returns A 128-bit integer vector containing the right-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sra_epi16(__m128i __a, __m128i __count)
{
  return (__m128i)__builtin_ia32_psraw128((__v8hi)__a, (__v8hi)__count);
}

/// \brief Right-shifts each 32-bit value in the 128-bit integer vector operand
///    by the specified number of bits. High-order bits are filled with the sign
///    bit of the initial value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSRAD / PSRAD instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    An integer value specifying the number of bits to right-shift each value
///    in operand __a.
/// \returns A 128-bit integer vector containing the right-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srai_epi32(__m128i __a, int __count)
{
  return (__m128i)__builtin_ia32_psradi128((__v4si)__a, __count);
}

/// \brief Right-shifts each 32-bit value in the 128-bit integer vector operand
///    by the specified number of bits. High-order bits are filled with the sign
///    bit of the initial value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSRAD / PSRAD instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    A 128-bit integer vector in which bits [63:0] specify the number of bits
///    to right-shift each value in operand __a.
/// \returns A 128-bit integer vector containing the right-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_sra_epi32(__m128i __a, __m128i __count)
{
  return (__m128i)__builtin_ia32_psrad128((__v4si)__a, (__v4si)__count);
}

/// \brief Right-shifts the 128-bit integer vector operand by the specified
///    number of bytes. High-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// __m128i _mm_srli_si128(__m128i a, const int imm);
/// \endcode
///
/// This intrinsic corresponds to the \c VPSRLDQ / PSRLDQ instruction.
///
/// \param a
///    A 128-bit integer vector containing the source operand.
/// \param imm
///    An immediate value specifying the number of bytes to right-shift operand
///    a.
/// \returns A 128-bit integer vector containing the right-shifted value.
#define _mm_srli_si128(a, imm) __extension__ ({                              \
  (__m128i)__builtin_shufflevector(                                          \
                                 (__v16qi)(__m128i)(a),                      \
                                 (__v16qi)_mm_setzero_si128(),               \
                                 ((char)(imm)&0xF0) ? 16 : (char)(imm) + 0,  \
                                 ((char)(imm)&0xF0) ? 17 : (char)(imm) + 1,  \
                                 ((char)(imm)&0xF0) ? 18 : (char)(imm) + 2,  \
                                 ((char)(imm)&0xF0) ? 19 : (char)(imm) + 3,  \
                                 ((char)(imm)&0xF0) ? 20 : (char)(imm) + 4,  \
                                 ((char)(imm)&0xF0) ? 21 : (char)(imm) + 5,  \
                                 ((char)(imm)&0xF0) ? 22 : (char)(imm) + 6,  \
                                 ((char)(imm)&0xF0) ? 23 : (char)(imm) + 7,  \
                                 ((char)(imm)&0xF0) ? 24 : (char)(imm) + 8,  \
                                 ((char)(imm)&0xF0) ? 25 : (char)(imm) + 9,  \
                                 ((char)(imm)&0xF0) ? 26 : (char)(imm) + 10, \
                                 ((char)(imm)&0xF0) ? 27 : (char)(imm) + 11, \
                                 ((char)(imm)&0xF0) ? 28 : (char)(imm) + 12, \
                                 ((char)(imm)&0xF0) ? 29 : (char)(imm) + 13, \
                                 ((char)(imm)&0xF0) ? 30 : (char)(imm) + 14, \
                                 ((char)(imm)&0xF0) ? 31 : (char)(imm) + 15); })

#define _mm_bsrli_si128(a, imm) \
  _mm_srli_si128((a), (imm))

/// \brief Right-shifts each of 16-bit values in the 128-bit integer vector
///    operand by the specified number of bits. High-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSRLW / PSRLW instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    An integer value specifying the number of bits to right-shift each value
///    in operand __a.
/// \returns A 128-bit integer vector containing the right-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srli_epi16(__m128i __a, int __count)
{
  return (__m128i)__builtin_ia32_psrlwi128((__v8hi)__a, __count);
}

/// \brief Right-shifts each of 16-bit values in the 128-bit integer vector
///    operand by the specified number of bits. High-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSRLW / PSRLW instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    A 128-bit integer vector in which bits [63:0] specify the number of bits
///    to right-shift each value in operand __a.
/// \returns A 128-bit integer vector containing the right-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srl_epi16(__m128i __a, __m128i __count)
{
  return (__m128i)__builtin_ia32_psrlw128((__v8hi)__a, (__v8hi)__count);
}

/// \brief Right-shifts each of 32-bit values in the 128-bit integer vector
///    operand by the specified number of bits. High-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSRLD / PSRLD instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    An integer value specifying the number of bits to right-shift each value
///    in operand __a.
/// \returns A 128-bit integer vector containing the right-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srli_epi32(__m128i __a, int __count)
{
  return (__m128i)__builtin_ia32_psrldi128((__v4si)__a, __count);
}

/// \brief Right-shifts each of 32-bit values in the 128-bit integer vector
///    operand by the specified number of bits. High-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSRLD / PSRLD instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    A 128-bit integer vector in which bits [63:0] specify the number of bits
///    to right-shift each value in operand __a.
/// \returns A 128-bit integer vector containing the right-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srl_epi32(__m128i __a, __m128i __count)
{
  return (__m128i)__builtin_ia32_psrld128((__v4si)__a, (__v4si)__count);
}

/// \brief Right-shifts each of 64-bit values in the 128-bit integer vector
///    operand by the specified number of bits. High-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSRLQ / PSRLQ instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    An integer value specifying the number of bits to right-shift each value
///    in operand __a.
/// \returns A 128-bit integer vector containing the right-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srli_epi64(__m128i __a, int __count)
{
  return __builtin_ia32_psrlqi128((__v2di)__a, __count);
}

/// \brief Right-shifts each of 64-bit values in the 128-bit integer vector
///    operand by the specified number of bits. High-order bits are cleared.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPSRLQ / PSRLQ instruction.
///
/// \param __a
///    A 128-bit integer vector containing the source operand.
/// \param __count
///    A 128-bit integer vector in which bits [63:0] specify the number of bits
///    to right-shift each value in operand __a.
/// \returns A 128-bit integer vector containing the right-shifted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_srl_epi64(__m128i __a, __m128i __count)
{
  return __builtin_ia32_psrlq128((__v2di)__a, (__v2di)__count);
}

/// \brief Compares each of the corresponding 8-bit values of the 128-bit
///    integer vectors for equality. Each comparison yields 0h for false, FFh
///    for true.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPCMPEQB / PCMPEQB instruction.
///
/// \param __a
///    A 128-bit integer vector.
/// \param __b
///    A 128-bit integer vector.
/// \returns A 128-bit integer vector containing the comparison results.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpeq_epi8(__m128i __a, __m128i __b)
{
  return (__m128i)((__v16qi)__a == (__v16qi)__b);
}

/// \brief Compares each of the corresponding 16-bit values of the 128-bit
///    integer vectors for equality. Each comparison yields 0h for false, FFFFh
///    for true.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPCMPEQW / PCMPEQW instruction.
///
/// \param __a
///    A 128-bit integer vector.
/// \param __b
///    A 128-bit integer vector.
/// \returns A 128-bit integer vector containing the comparison results.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpeq_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)((__v8hi)__a == (__v8hi)__b);
}

/// \brief Compares each of the corresponding 32-bit values of the 128-bit
///    integer vectors for equality. Each comparison yields 0h for false,
///    FFFFFFFFh for true.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPCMPEQD / PCMPEQD instruction.
///
/// \param __a
///    A 128-bit integer vector.
/// \param __b
///    A 128-bit integer vector.
/// \returns A 128-bit integer vector containing the comparison results.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpeq_epi32(__m128i __a, __m128i __b)
{
  return (__m128i)((__v4si)__a == (__v4si)__b);
}

/// \brief Compares each of the corresponding signed 8-bit values of the 128-bit
///    integer vectors to determine if the values in the first operand are
///    greater than those in the second operand. Each comparison yields 0h for
///    false, FFh for true.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPCMPGTB / PCMPGTB instruction.
///
/// \param __a
///    A 128-bit integer vector.
/// \param __b
///    A 128-bit integer vector.
/// \returns A 128-bit integer vector containing the comparison results.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpgt_epi8(__m128i __a, __m128i __b)
{
  /* This function always performs a signed comparison, but __v16qi is a char
     which may be signed or unsigned, so use __v16qs. */
  return (__m128i)((__v16qs)__a > (__v16qs)__b);
}

/// \brief Compares each of the corresponding signed 16-bit values of the
///    128-bit integer vectors to determine if the values in the first operand
///    are greater than those in the second operand. Each comparison yields 0h
///    for false, FFFFh for true.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPCMPGTW / PCMPGTW instruction.
///
/// \param __a
///    A 128-bit integer vector.
/// \param __b
///    A 128-bit integer vector.
/// \returns A 128-bit integer vector containing the comparison results.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpgt_epi16(__m128i __a, __m128i __b)
{
  return (__m128i)((__v8hi)__a > (__v8hi)__b);
}

/// \brief Compares each of the corresponding signed 32-bit values of the
///    128-bit integer vectors to determine if the values in the first operand
///    are greater than those in the second operand. Each comparison yields 0h
///    for false, FFFFFFFFh for true.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPCMPGTD / PCMPGTD instruction.
///
/// \param __a
///    A 128-bit integer vector.
/// \param __b
///    A 128-bit integer vector.
/// \returns A 128-bit integer vector containing the comparison results.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpgt_epi32(__m128i __a, __m128i __b)
{
  return (__m128i)((__v4si)__a > (__v4si)__b);
}

/// \brief Compares each of the corresponding signed 8-bit values of the 128-bit
///    integer vectors to determine if the values in the first operand are less
///    than those in the second operand. Each comparison yields 0h for false,
///    FFh for true.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPCMPGTB / PCMPGTB instruction.
///
/// \param __a
///    A 128-bit integer vector.
/// \param __b
///    A 128-bit integer vector.
/// \returns A 128-bit integer vector containing the comparison results.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmplt_epi8(__m128i __a, __m128i __b)
{
  return _mm_cmpgt_epi8(__b, __a);
}

/// \brief Compares each of the corresponding signed 16-bit values of the
///    128-bit integer vectors to determine if the values in the first operand
///    are less than those in the second operand. Each comparison yields 0h for
///    false, FFFFh for true.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPCMPGTW / PCMPGTW instruction.
///
/// \param __a
///    A 128-bit integer vector.
/// \param __b
///    A 128-bit integer vector.
/// \returns A 128-bit integer vector containing the comparison results.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmplt_epi16(__m128i __a, __m128i __b)
{
  return _mm_cmpgt_epi16(__b, __a);
}

/// \brief Compares each of the corresponding signed 32-bit values of the
///    128-bit integer vectors to determine if the values in the first operand
///    are less than those in the second operand. Each comparison yields 0h for
///    false, FFFFFFFFh for true.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VPCMPGTD / PCMPGTD instruction.
///
/// \param __a
///    A 128-bit integer vector.
/// \param __b
///    A 128-bit integer vector.
/// \returns A 128-bit integer vector containing the comparison results.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmplt_epi32(__m128i __a, __m128i __b)
{
  return _mm_cmpgt_epi32(__b, __a);
}

#ifdef __x86_64__
/// \brief Converts a 64-bit signed integer value from the second operand into a
///    double-precision value and returns it in the lower element of a [2 x
///    double] vector; the upper element of the returned vector is copied from
///    the upper element of the first operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCVTSI2SD / CVTSI2SD instruction.
///
/// \param __a
///    A 128-bit vector of [2 x double]. The upper 64 bits of this operand are
///    copied to the upper 64 bits of the destination.
/// \param __b
///    A 64-bit signed integer operand containing the value to be converted.
/// \returns A 128-bit vector of [2 x double] whose lower 64 bits contain the
///    converted value of the second operand. The upper 64 bits are copied from
///    the upper 64 bits of the first operand.
static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_cvtsi64_sd(__m128d __a, long long __b)
{
  __a[0] = __b;
  return __a;
}

/// \brief Converts the first (lower) element of a vector of [2 x double] into a
///    64-bit signed integer value, according to the current rounding mode.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCVTSD2SI / CVTSD2SI instruction.
///
/// \param __a
///    A 128-bit vector of [2 x double]. The lower 64 bits are used in the
///    conversion.
/// \returns A 64-bit signed integer containing the converted value.
static __inline__ long long __DEFAULT_FN_ATTRS
_mm_cvtsd_si64(__m128d __a)
{
  return __builtin_ia32_cvtsd2si64((__v2df)__a);
}

/// \brief Converts the first (lower) element of a vector of [2 x double] into a
///    64-bit signed integer value, truncating the result when it is inexact.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCVTTSD2SI / CVTTSD2SI instruction.
///
/// \param __a
///    A 128-bit vector of [2 x double]. The lower 64 bits are used in the
///    conversion.
/// \returns A 64-bit signed integer containing the converted value.
static __inline__ long long __DEFAULT_FN_ATTRS
_mm_cvttsd_si64(__m128d __a)
{
  return __a[0];
}
#endif

/// \brief Converts a vector of [4 x i32] into a vector of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCVTDQ2PS / CVTDQ2PS instruction.
///
/// \param __a
///    A 128-bit integer vector.
/// \returns A 128-bit vector of [4 x float] containing the converted values.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtepi32_ps(__m128i __a)
{
  return __builtin_ia32_cvtdq2ps((__v4si)__a);
}

/// \brief Converts a vector of [4 x float] into a vector of [4 x i32].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2DQ / CVTPS2DQ instruction.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit integer vector of [4 x i32] containing the converted
///    values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvtps_epi32(__m128 __a)
{
  return (__m128i)__builtin_ia32_cvtps2dq((__v4sf)__a);
}

/// \brief Converts a vector of [4 x float] into a vector of [4 x i32],
///    truncating the result when it is inexact.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCVTTPS2DQ / CVTTPS2DQ instruction.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x i32] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvttps_epi32(__m128 __a)
{
  return (__m128i)__builtin_convertvector((__v4sf)__a, __v4si);
}

/// \brief Returns a vector of [4 x i32] where the lowest element is the input
///    operand and the remaining elements are zero.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VMOVD / MOVD instruction.
///
/// \param __a
///    A 32-bit signed integer operand.
/// \returns A 128-bit vector of [4 x i32].
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvtsi32_si128(int __a)
{
  return (__m128i)(__v4si){ __a, 0, 0, 0 };
}

#ifdef __x86_64__
/// \brief Returns a vector of [2 x i64] where the lower element is the input
///    operand and the upper element is zero.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VMOVQ / MOVQ instruction.
///
/// \param __a
///    A 64-bit signed integer operand containing the value to be converted.
/// \returns A 128-bit vector of [2 x i64] containing the converted value.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cvtsi64_si128(long long __a)
{
  return (__m128i){ __a, 0 };
}
#endif

/// \brief Moves the least significant 32 bits of a vector of [4 x i32] to a
///    32-bit signed integer value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VMOVD / MOVD instruction.
///
/// \param __a
///    A vector of [4 x i32]. The least significant 32 bits are moved to the
///    destination.
/// \returns A 32-bit signed integer containing the moved value.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_cvtsi128_si32(__m128i __a)
{
  __v4si __b = (__v4si)__a;
  return __b[0];
}

#ifdef __x86_64__
/// \brief Moves the least significant 64 bits of a vector of [2 x i64] to a
///    64-bit signed integer value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VMOVQ / MOVQ instruction.
///
/// \param __a
///    A vector of [2 x i64]. The least significant 64 bits are moved to the
///    destination.
/// \returns A 64-bit signed integer containing the moved value.
static __inline__ long long __DEFAULT_FN_ATTRS
_mm_cvtsi128_si64(__m128i __a)
{
  return __a[0];
}
#endif

/// \brief Moves packed integer values from an aligned 128-bit memory location
///    to elements in a 128-bit integer vector.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VMOVDQA / MOVDQA instruction.
///
/// \param __p
///    An aligned pointer to a memory location containing integer values.
/// \returns A 128-bit integer vector containing the moved values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_load_si128(__m128i const *__p)
{
  return *__p;
}

/// \brief Moves packed integer values from an unaligned 128-bit memory location
///    to elements in a 128-bit integer vector.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VMOVDQU / MOVDQU instruction.
///
/// \param __p
///    A pointer to a memory location containing integer values.
/// \returns A 128-bit integer vector containing the moved values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_loadu_si128(__m128i const *__p)
{
  struct __loadu_si128 {
    __m128i __v;
  } __attribute__((__packed__, __may_alias__));
  return ((struct __loadu_si128*)__p)->__v;
}

/// \brief Returns a vector of [2 x i64] where the lower element is taken from
///    the lower element of the operand, and the upper element is zero.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VMOVQ / MOVQ instruction.
///
/// \param __p
///    A 128-bit vector of [2 x i64]. Bits [63:0] are written to bits [63:0] of
///    the destination.
/// \returns A 128-bit vector of [2 x i64]. The lower order bits contain the
///    moved value. The higher order bits are cleared.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_loadl_epi64(__m128i const *__p)
{
  struct __mm_loadl_epi64_struct {
    long long __u;
  } __attribute__((__packed__, __may_alias__));
  return (__m128i) { ((struct __mm_loadl_epi64_struct*)__p)->__u, 0};
}

/// \brief Generates a 128-bit vector of [4 x i32] with unspecified content.
///    This could be used as an argument to another intrinsic function where the
///    argument is required but the value is not actually used.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic has no corresponding instruction.
///
/// \returns A 128-bit vector of [4 x i32] with unspecified content.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_undefined_si128(void)
{
  return (__m128i)__builtin_ia32_undef128();
}

/// \brief Initializes both 64-bit values in a 128-bit vector of [2 x i64] with
///    the specified 64-bit integer values.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __q1
///    A 64-bit integer value used to initialize the upper 64 bits of the
///    destination vector of [2 x i64].
/// \param __q0
///    A 64-bit integer value used to initialize the lower 64 bits of the
///    destination vector of [2 x i64].
/// \returns An initialized 128-bit vector of [2 x i64] containing the values
///    provided in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set_epi64x(long long __q1, long long __q0)
{
  return (__m128i){ __q0, __q1 };
}

/// \brief Initializes both 64-bit values in a 128-bit vector of [2 x i64] with
///    the specified 64-bit integer values.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __q1
///    A 64-bit integer value used to initialize the upper 64 bits of the
///    destination vector of [2 x i64].
/// \param __q0
///    A 64-bit integer value used to initialize the lower 64 bits of the
///    destination vector of [2 x i64].
/// \returns An initialized 128-bit vector of [2 x i64] containing the values
///    provided in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set_epi64(__m64 __q1, __m64 __q0)
{
  return (__m128i){ (long long)__q0, (long long)__q1 };
}

/// \brief Initializes the 32-bit values in a 128-bit vector of [4 x i32] with
///    the specified 32-bit integer values.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __i3
///    A 32-bit integer value used to initialize bits [127:96] of the
///    destination vector.
/// \param __i2
///    A 32-bit integer value used to initialize bits [95:64] of the destination
///    vector.
/// \param __i1
///    A 32-bit integer value used to initialize bits [63:32] of the destination
///    vector.
/// \param __i0
///    A 32-bit integer value used to initialize bits [31:0] of the destination
///    vector.
/// \returns An initialized 128-bit vector of [4 x i32] containing the values
///    provided in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set_epi32(int __i3, int __i2, int __i1, int __i0)
{
  return (__m128i)(__v4si){ __i0, __i1, __i2, __i3};
}

/// \brief Initializes the 16-bit values in a 128-bit vector of [8 x i16] with
///    the specified 16-bit integer values.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __w7
///    A 16-bit integer value used to initialize bits [127:112] of the
///    destination vector.
/// \param __w6
///    A 16-bit integer value used to initialize bits [111:96] of the
///    destination vector.
/// \param __w5
///    A 16-bit integer value used to initialize bits [95:80] of the destination
///    vector.
/// \param __w4
///    A 16-bit integer value used to initialize bits [79:64] of the destination
///    vector.
/// \param __w3
///    A 16-bit integer value used to initialize bits [63:48] of the destination
///    vector.
/// \param __w2
///    A 16-bit integer value used to initialize bits [47:32] of the destination
///    vector.
/// \param __w1
///    A 16-bit integer value used to initialize bits [31:16] of the destination
///    vector.
/// \param __w0
///    A 16-bit integer value used to initialize bits [15:0] of the destination
///    vector.
/// \returns An initialized 128-bit vector of [8 x i16] containing the values
///    provided in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set_epi16(short __w7, short __w6, short __w5, short __w4, short __w3, short __w2, short __w1, short __w0)
{
  return (__m128i)(__v8hi){ __w0, __w1, __w2, __w3, __w4, __w5, __w6, __w7 };
}

/// \brief Initializes the 8-bit values in a 128-bit vector of [16 x i8] with
///    the specified 8-bit integer values.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __b15
///    Initializes bits [127:120] of the destination vector.
/// \param __b14
///    Initializes bits [119:112] of the destination vector.
/// \param __b13
///    Initializes bits [111:104] of the destination vector.
/// \param __b12
///    Initializes bits [103:96] of the destination vector.
/// \param __b11
///    Initializes bits [95:88] of the destination vector.
/// \param __b10
///    Initializes bits [87:80] of the destination vector.
/// \param __b9
///    Initializes bits [79:72] of the destination vector.
/// \param __b8
///    Initializes bits [71:64] of the destination vector.
/// \param __b7
///    Initializes bits [63:56] of the destination vector.
/// \param __b6
///    Initializes bits [55:48] of the destination vector.
/// \param __b5
///    Initializes bits [47:40] of the destination vector.
/// \param __b4
///    Initializes bits [39:32] of the destination vector.
/// \param __b3
///    Initializes bits [31:24] of the destination vector.
/// \param __b2
///    Initializes bits [23:16] of the destination vector.
/// \param __b1
///    Initializes bits [15:8] of the destination vector.
/// \param __b0
///    Initializes bits [7:0] of the destination vector.
/// \returns An initialized 128-bit vector of [16 x i8] containing the values
///    provided in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set_epi8(char __b15, char __b14, char __b13, char __b12, char __b11, char __b10, char __b9, char __b8, char __b7, char __b6, char __b5, char __b4, char __b3, char __b2, char __b1, char __b0)
{
  return (__m128i)(__v16qi){ __b0, __b1, __b2, __b3, __b4, __b5, __b6, __b7, __b8, __b9, __b10, __b11, __b12, __b13, __b14, __b15 };
}

/// \brief Initializes both values in a 128-bit integer vector with the
///    specified 64-bit integer value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __q
///    Integer value used to initialize the elements of the destination integer
///    vector.
/// \returns An initialized 128-bit integer vector of [2 x i64] with both
///    elements containing the value provided in the operand.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set1_epi64x(long long __q)
{
  return (__m128i){ __q, __q };
}

/// \brief Initializes both values in a 128-bit vector of [2 x i64] with the
///    specified 64-bit value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __q
///    A 64-bit value used to initialize the elements of the destination integer
///    vector.
/// \returns An initialized 128-bit vector of [2 x i64] with all elements
///    containing the value provided in the operand.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set1_epi64(__m64 __q)
{
  return (__m128i){ (long long)__q, (long long)__q };
}

/// \brief Initializes all values in a 128-bit vector of [4 x i32] with the
///    specified 32-bit value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __i
///    A 32-bit value used to initialize the elements of the destination integer
///    vector.
/// \returns An initialized 128-bit vector of [4 x i32] with all elements
///    containing the value provided in the operand.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set1_epi32(int __i)
{
  return (__m128i)(__v4si){ __i, __i, __i, __i };
}

/// \brief Initializes all values in a 128-bit vector of [8 x i16] with the
///    specified 16-bit value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __w
///    A 16-bit value used to initialize the elements of the destination integer
///    vector.
/// \returns An initialized 128-bit vector of [8 x i16] with all elements
///    containing the value provided in the operand.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set1_epi16(short __w)
{
  return (__m128i)(__v8hi){ __w, __w, __w, __w, __w, __w, __w, __w };
}

/// \brief Initializes all values in a 128-bit vector of [16 x i8] with the
///    specified 8-bit value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __b
///    An 8-bit value used to initialize the elements of the destination integer
///    vector.
/// \returns An initialized 128-bit vector of [16 x i8] with all elements
///    containing the value provided in the operand.
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
  struct __storeu_si128 {
    __m128i __v;
  } __attribute__((__packed__, __may_alias__));
  ((struct __storeu_si128*)__p)->__v = __b;
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
  __builtin_nontemporal_store((__v2df)__a, (__v2df*)__p);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_stream_si128(__m128i *__p, __m128i __a)
{
  __builtin_nontemporal_store((__v2di)__a, (__v2di*)__p);
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
                                   (__v4si)_mm_undefined_si128(), \
                                   ((imm) >> 0) & 0x3, ((imm) >> 2) & 0x3, \
                                   ((imm) >> 4) & 0x3, ((imm) >> 6) & 0x3); })

#define _mm_shufflelo_epi16(a, imm) __extension__ ({ \
  (__m128i)__builtin_shufflevector((__v8hi)(__m128i)(a), \
                                   (__v8hi)_mm_undefined_si128(), \
                                   ((imm) >> 0) & 0x3, ((imm) >> 2) & 0x3, \
                                   ((imm) >> 4) & 0x3, ((imm) >> 6) & 0x3, \
                                   4, 5, 6, 7); })

#define _mm_shufflehi_epi16(a, imm) __extension__ ({ \
  (__m128i)__builtin_shufflevector((__v8hi)(__m128i)(a), \
                                   (__v8hi)_mm_undefined_si128(), \
                                   0, 1, 2, 3, \
                                   4 + (((imm) >> 0) & 0x3), \
                                   4 + (((imm) >> 2) & 0x3), \
                                   4 + (((imm) >> 4) & 0x3), \
                                   4 + (((imm) >> 6) & 0x3)); })

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
  return (__m128i)__builtin_shufflevector((__v2di)__a, (__v2di)__b, 1, 2+1);
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
  return (__m128i)__builtin_shufflevector((__v2di)__a, (__v2di)__b, 0, 2+0);
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
  return __builtin_shufflevector((__v2di)__a, (__m128i){ 0 }, 0, 2);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_unpackhi_pd(__m128d __a, __m128d __b)
{
  return __builtin_shufflevector((__v2df)__a, (__v2df)__b, 1, 2+1);
}

static __inline__ __m128d __DEFAULT_FN_ATTRS
_mm_unpacklo_pd(__m128d __a, __m128d __b)
{
  return __builtin_shufflevector((__v2df)__a, (__v2df)__b, 0, 2+0);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_movemask_pd(__m128d __a)
{
  return __builtin_ia32_movmskpd((__v2df)__a);
}

#define _mm_shuffle_pd(a, b, i) __extension__ ({ \
  (__m128d)__builtin_shufflevector((__v2df)(__m128d)(a), (__v2df)(__m128d)(b), \
                                   0 + (((i) >> 0) & 0x1), \
                                   2 + (((i) >> 1) & 0x1)); })

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
