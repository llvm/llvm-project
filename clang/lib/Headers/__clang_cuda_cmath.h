/*===---- __clang_cuda_cmath.h - Device-side CUDA cmath support ------------===
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
#ifndef __CLANG_CUDA_CMATH_H__
#define __CLANG_CUDA_CMATH_H__
#ifndef __CUDA__
#error "This file is for CUDA compilation only."
#endif

// CUDA allows using math functions form std:: on device side.  This
// file provides __device__ overloads for math functions that map to
// appropriate math functions provided by CUDA headers or to compiler
// builtins if CUDA does not provide a suitable function.

#define __DEVICE__ static __device__ __inline__ __attribute__((always_inline))

namespace std {
__DEVICE__ long long abs(long long n) { return ::llabs(n); }
__DEVICE__ long abs(long n) { return ::labs(n); }
__DEVICE__ int abs(int n) { return ::abs(n); }
__DEVICE__ float abs(float x) { return ::fabsf(x); }
__DEVICE__ double abs(double x) { return ::fabs(x); }
__DEVICE__ float acos(float x) { return ::acosf(x); }
__DEVICE__ double acos(double x) { return ::acos(x); }
__DEVICE__ float acosh(float x) { return ::acoshf(x); }
__DEVICE__ double acosh(double x) { return ::acosh(x); }
__DEVICE__ float asin(float x) { return ::asinf(x); }
__DEVICE__ double asin(double x) { return ::asin(x); }
__DEVICE__ float asinh(float x) { return ::asinhf(x); }
__DEVICE__ double asinh(double x) { return ::asinh(x); }
__DEVICE__ float atan(float x) { return ::atanf(x); }
__DEVICE__ double atan(double x) { return ::atan(x); }
__DEVICE__ float atan2(float x, float y) { return ::atan2f(x, y); }
__DEVICE__ double atan2(double x, double y) { return ::atan2(x, y); }
__DEVICE__ float atanh(float x) { return ::atanhf(x); }
__DEVICE__ double atanh(double x) { return ::atanh(x); }
__DEVICE__ float cbrt(float x) { return ::cbrtf(x); }
__DEVICE__ double cbrt(double x) { return ::cbrt(x); }
__DEVICE__ float ceil(float x) { return ::ceilf(x); }
__DEVICE__ double ceil(double x) { return ::ceil(x); }
__DEVICE__ float copysign(float x, float y) { return ::copysignf(x, y); }
__DEVICE__ double copysign(double x, double y) { return ::copysign(x, y); }
__DEVICE__ float cos(float x) { return ::cosf(x); }
__DEVICE__ double cos(double x) { return ::cos(x); }
__DEVICE__ float cosh(float x) { return ::coshf(x); }
__DEVICE__ double cosh(double x) { return ::cosh(x); }
__DEVICE__ float erf(float x) { return ::erff(x); }
__DEVICE__ double erf(double x) { return ::erf(x); }
__DEVICE__ float erfc(float x) { return ::erfcf(x); }
__DEVICE__ double erfc(double x) { return ::erfc(x); }
__DEVICE__ float exp(float x) { return ::expf(x); }
__DEVICE__ double exp(double x) { return ::exp(x); }
__DEVICE__ float exp2(float x) { return ::exp2f(x); }
__DEVICE__ double exp2(double x) { return ::exp2(x); }
__DEVICE__ float expm1(float x) { return ::expm1f(x); }
__DEVICE__ double expm1(double x) { return ::expm1(x); }
__DEVICE__ float fabs(float x) { return ::fabsf(x); }
__DEVICE__ double fabs(double x) { return ::fabs(x); }
__DEVICE__ float fdim(float x, float y) { return ::fdimf(x, y); }
__DEVICE__ double fdim(double x, double y) { return ::fdim(x, y); }
__DEVICE__ float floor(float x) { return ::floorf(x); }
__DEVICE__ double floor(double x) { return ::floor(x); }
__DEVICE__ float fma(float x, float y, float z) { return ::fmaf(x, y, z); }
__DEVICE__ double fma(double x, double y, double z) { return ::fma(x, y, z); }
__DEVICE__ float fmax(float x, float y) { return ::fmaxf(x, y); }
__DEVICE__ double fmax(double x, double y) { return ::fmax(x, y); }
__DEVICE__ float fmin(float x, float y) { return ::fminf(x, y); }
__DEVICE__ double fmin(double x, double y) { return ::fmin(x, y); }
__DEVICE__ float fmod(float x, float y) { return ::fmodf(x, y); }
__DEVICE__ double fmod(double x, double y) { return ::fmod(x, y); }
__DEVICE__ int fpclassify(float x) {
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL,
                              FP_ZERO, x);
}
__DEVICE__ int fpclassify(double x) {
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL,
                              FP_ZERO, x);
}
__DEVICE__ float frexp(float arg, int *exp) { return ::frexpf(arg, exp); }
__DEVICE__ double frexp(double arg, int *exp) { return ::frexp(arg, exp); }
__DEVICE__ float hypot(float x, float y) { return ::hypotf(x, y); }
__DEVICE__ double hypot(double x, double y) { return ::hypot(x, y); }
__DEVICE__ int ilogb(float arg) { return ::ilogbf(arg); }
__DEVICE__ int ilogb(double arg) { return ::ilogb(arg); }
__DEVICE__ bool isfinite(float x) { return ::__finitef(x); }
__DEVICE__ bool isfinite(double x) { return ::__finite(x); }
__DEVICE__ bool isgreater(float x, float y) {
  return __builtin_isgreater(x, y);
}
__DEVICE__ bool isgreater(double x, double y) {
  return __builtin_isgreater(x, y);
}
__DEVICE__ bool isgreaterequal(float x, float y) {
  return __builtin_isgreaterequal(x, y);
}
__DEVICE__ bool isgreaterequal(double x, double y) {
  return __builtin_isgreaterequal(x, y);
}
__DEVICE__ bool isinf(float x) { return ::__isinff(x); }
__DEVICE__ bool isinf(double x) { return ::__isinf(x); }
__DEVICE__ bool isless(float x, float y) { return __builtin_isless(x, y); }
__DEVICE__ bool isless(double x, double y) { return __builtin_isless(x, y); }
__DEVICE__ bool islessequal(float x, float y) {
  return __builtin_islessequal(x, y);
}
__DEVICE__ bool islessequal(double x, double y) {
  return __builtin_islessequal(x, y);
}
__DEVICE__ bool islessgreater(float x, float y) {
  return __builtin_islessgreater(x, y);
}
__DEVICE__ bool islessgreater(double x, double y) {
  return __builtin_islessgreater(x, y);
}
__DEVICE__ bool isnan(float x) { return ::__isnanf(x); }
__DEVICE__ bool isnan(double x) { return ::__isnan(x); }
__DEVICE__ bool isnormal(float x) { return __builtin_isnormal(x); }
__DEVICE__ bool isnormal(double x) { return __builtin_isnormal(x); }
__DEVICE__ bool isunordered(float x, float y) {
  return __builtin_isunordered(x, y);
}
__DEVICE__ bool isunordered(double x, double y) {
  return __builtin_isunordered(x, y);
}
__DEVICE__ long labs(long n) { return ::labs(n); }
__DEVICE__ float ldexp(float arg, int exp) { return ::ldexpf(arg, exp); }
__DEVICE__ double ldexp(double arg, int exp) { return ::ldexp(arg, exp); }
__DEVICE__ float lgamma(float x) { return ::lgammaf(x); }
__DEVICE__ double lgamma(double x) { return ::lgamma(x); }
__DEVICE__ long long llabs(long long n) { return ::llabs(n); }
__DEVICE__ long long llrint(float x) { return ::llrintf(x); }
__DEVICE__ long long llrint(double x) { return ::llrint(x); }
__DEVICE__ float log(float x) { return ::logf(x); }
__DEVICE__ double log(double x) { return ::log(x); }
__DEVICE__ float log10(float x) { return ::log10f(x); }
__DEVICE__ double log10(double x) { return ::log10(x); }
__DEVICE__ float log1p(float x) { return ::log1pf(x); }
__DEVICE__ double log1p(double x) { return ::log1p(x); }
__DEVICE__ float log2(float x) { return ::log2f(x); }
__DEVICE__ double log2(double x) { return ::log2(x); }
__DEVICE__ float logb(float x) { return ::logbf(x); }
__DEVICE__ double logb(double x) { return ::logb(x); }
__DEVICE__ long lrint(float x) { return ::lrintf(x); }
__DEVICE__ long lrint(double x) { return ::lrint(x); }
__DEVICE__ long lround(float x) { return ::lroundf(x); }
__DEVICE__ long lround(double x) { return ::lround(x); }
__DEVICE__ float modf(float x, float *iptr) { return ::modff(x, iptr); }
__DEVICE__ double modf(double x, double *iptr) { return ::modf(x, iptr); }
__DEVICE__ double nan(const char *x) { return ::nan(x); }
__DEVICE__ float nanf(const char *x) { return ::nanf(x); }
__DEVICE__ float nearbyint(float x) { return ::nearbyintf(x); }
__DEVICE__ double nearbyint(double x) { return ::nearbyint(x); }
__DEVICE__ float nextafter(float from, float to) {
  return ::nextafterf(from, to);
}
__DEVICE__ double nextafter(double from, double to) {
  return ::nextafter(from, to);
}
__DEVICE__ float nexttoward(float from, float to) {
  return __builtin_nexttowardf(from, to);
}
__DEVICE__ double nexttoward(double from, double to) {
  return __builtin_nexttoward(from, to);
}
__DEVICE__ float pow(float base, float exp) { return ::powf(base, exp); }
__DEVICE__ float pow(float base, int iexp) { return ::powif(base, iexp); }
__DEVICE__ double pow(double base, double exp) { return ::pow(base, exp); }
__DEVICE__ double pow(double base, int iexp) { return ::powi(base, iexp); }
__DEVICE__ float remainder(float x, float y) { return ::remainderf(x, y); }
__DEVICE__ double remainder(double x, double y) { return ::remainder(x, y); }
__DEVICE__ float remquo(float x, float y, int *quo) {
  return ::remquof(x, y, quo);
}
__DEVICE__ double remquo(double x, double y, int *quo) {
  return ::remquo(x, y, quo);
}
__DEVICE__ float rint(float x) { return ::rintf(x); }
__DEVICE__ double rint(double x) { return ::rint(x); }
__DEVICE__ float round(float x) { return ::roundf(x); }
__DEVICE__ double round(double x) { return ::round(x); }
__DEVICE__ float scalbln(float x, long exp) { return ::scalblnf(x, exp); }
__DEVICE__ double scalbln(double x, long exp) { return ::scalbln(x, exp); }
__DEVICE__ float scalbn(float x, int exp) { return ::scalbnf(x, exp); }
__DEVICE__ double scalbn(double x, int exp) { return ::scalbn(x, exp); }
__DEVICE__ bool signbit(float x) { return ::__signbitf(x); }
__DEVICE__ bool signbit(double x) { return ::__signbit(x); }
__DEVICE__ float sin(float x) { return ::sinf(x); }
__DEVICE__ double sin(double x) { return ::sin(x); }
__DEVICE__ float sinh(float x) { return ::sinhf(x); }
__DEVICE__ double sinh(double x) { return ::sinh(x); }
__DEVICE__ float sqrt(float x) { return ::sqrtf(x); }
__DEVICE__ double sqrt(double x) { return ::sqrt(x); }
__DEVICE__ float tan(float x) { return ::tanf(x); }
__DEVICE__ double tan(double x) { return ::tan(x); }
__DEVICE__ float tanh(float x) { return ::tanhf(x); }
__DEVICE__ double tanh(double x) { return ::tanh(x); }
__DEVICE__ float tgamma(float x) { return ::tgammaf(x); }
__DEVICE__ double tgamma(double x) { return ::tgamma(x); }
__DEVICE__ float trunc(float x) { return ::truncf(x); }
__DEVICE__ double trunc(double x) { return ::trunc(x); }

} // namespace std

#endif
