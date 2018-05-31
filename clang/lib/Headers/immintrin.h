/*===---- immintrin.h - Intel intrinsics -----------------------------------===
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
#define __IMMINTRIN_H

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__MMX__)
#include <mmintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__SSE__)
#include <xmmintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__SSE2__)
#include <emmintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__SSE3__)
#include <pmmintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__SSSE3__)
#include <tmmintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || \
    (defined(__SSE4_2__) || defined(__SSE4_1__))
#include <smmintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || \
    (defined(__AES__) || defined(__PCLMUL__))
#include <wmmintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__CLFLUSHOPT__)
#include <clflushoptintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__CLWB__)
#include <clwbintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX__)
#include <avxintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX2__)
#include <avx2intrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__F16C__)
#include <f16cintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__VPCLMULQDQ__)
#include <vpclmulqdqintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__BMI__)
#include <bmiintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__BMI2__)
#include <bmi2intrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__LZCNT__)
#include <lzcntintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__POPCNT__)
#include <popcntintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__FMA__)
#include <fmaintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX512F__)
#include <avx512fintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX512VL__)
#include <avx512vlintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX512BW__)
#include <avx512bwintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX512BITALG__)
#include <avx512bitalgintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX512CD__)
#include <avx512cdintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX512VPOPCNTDQ__)
#include <avx512vpopcntdqintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || \
    (defined(__AVX512VL__) && defined(__AVX512VPOPCNTDQ__))
#include <avx512vpopcntdqvlintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX512VNNI__)
#include <avx512vnniintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || \
    (defined(__AVX512VL__) && defined(__AVX512VNNI__))
#include <avx512vlvnniintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX512DQ__)
#include <avx512dqintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || \
    (defined(__AVX512VL__) && defined(__AVX512BITALG__))
#include <avx512vlbitalgintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || \
    (defined(__AVX512VL__) && defined(__AVX512BW__))
#include <avx512vlbwintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || \
    (defined(__AVX512VL__) && defined(__AVX512CD__))
#include <avx512vlcdintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || \
    (defined(__AVX512VL__) && defined(__AVX512DQ__))
#include <avx512vldqintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX512ER__)
#include <avx512erintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX512IFMA__)
#include <avx512ifmaintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || \
    (defined(__AVX512IFMA__) && defined(__AVX512VL__))
#include <avx512ifmavlintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX512VBMI__)
#include <avx512vbmiintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || \
    (defined(__AVX512VBMI__) && defined(__AVX512VL__))
#include <avx512vbmivlintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX512VBMI2__)
#include <avx512vbmi2intrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || \
    (defined(__AVX512VBMI2__) && defined(__AVX512VL__))
#include <avx512vlvbmi2intrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__AVX512PF__)
#include <avx512pfintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__PKU__)
#include <pkuintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__VAES__)
#include <vaesintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__GFNI__)
#include <gfniintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__RDPID__)
/// Returns the value of the IA32_TSC_AUX MSR (0xc0000103).
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> RDPID </c> instruction.
static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__, __target__("rdpid")))
_rdpid_u32(void) {
  return __builtin_ia32_rdpid();
}
#endif // __RDPID__

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__RDRND__)
static __inline__ int __attribute__((__always_inline__, __nodebug__, __target__("rdrnd")))
_rdrand16_step(unsigned short *__p)
{
  return __builtin_ia32_rdrand16_step(__p);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__, __target__("rdrnd")))
_rdrand32_step(unsigned int *__p)
{
  return __builtin_ia32_rdrand32_step(__p);
}

#ifdef __x86_64__
static __inline__ int __attribute__((__always_inline__, __nodebug__, __target__("rdrnd")))
_rdrand64_step(unsigned long long *__p)
{
  return __builtin_ia32_rdrand64_step(__p);
}
#endif
#endif /* __RDRND__ */

/* __bit_scan_forward */
static __inline__ int __attribute__((__always_inline__, __nodebug__))
_bit_scan_forward(int __A) {
  return __builtin_ctz(__A);
}

/* __bit_scan_reverse */
static __inline__ int __attribute__((__always_inline__, __nodebug__))
_bit_scan_reverse(int __A) {
  return 31 - __builtin_clz(__A);
}

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__FSGSBASE__)
#ifdef __x86_64__
static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_readfsbase_u32(void)
{
  return __builtin_ia32_rdfsbase32();
}

static __inline__ unsigned long long __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_readfsbase_u64(void)
{
  return __builtin_ia32_rdfsbase64();
}

static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_readgsbase_u32(void)
{
  return __builtin_ia32_rdgsbase32();
}

static __inline__ unsigned long long __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_readgsbase_u64(void)
{
  return __builtin_ia32_rdgsbase64();
}

static __inline__ void __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_writefsbase_u32(unsigned int __V)
{
  __builtin_ia32_wrfsbase32(__V);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_writefsbase_u64(unsigned long long __V)
{
  __builtin_ia32_wrfsbase64(__V);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_writegsbase_u32(unsigned int __V)
{
  __builtin_ia32_wrgsbase32(__V);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_writegsbase_u64(unsigned long long __V)
{
  __builtin_ia32_wrgsbase64(__V);
}

#endif
#endif /* __FSGSBASE__ */

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__RTM__)
#include <rtmintrin.h>
#include <xtestintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__SHA__)
#include <shaintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__FXSR__)
#include <fxsrintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__XSAVE__)
#include <xsaveintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__XSAVEOPT__)
#include <xsaveoptintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__XSAVEC__)
#include <xsavecintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__XSAVES__)
#include <xsavesintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__SHSTK__)
#include <cetintrin.h>
#endif

/* Some intrinsics inside adxintrin.h are available only on processors with ADX,
 * whereas others are also available at all times. */
#include <adxintrin.h>

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__RDSEED__)
#include <rdseedintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__WBNOINVD__)
#include <wbnoinvdintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__CLDEMOTE__)
#include <cldemoteintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__WAITPKG__)
#include <waitpkgintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || \
  defined(__MOVDIRI__) || defined(__MOVDIR64B__)
#include <movdirintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__PCONFIG__)
#include <pconfigintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__SGX__)
#include <sgxintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__PTWRITE__)
#include <ptwriteintrin.h>
#endif

#if !defined(_MSC_VER) || __has_feature(modules) || defined(__INVPCID__)
#include <invpcidintrin.h>
#endif

#endif /* __IMMINTRIN_H */
