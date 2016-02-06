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
using ::abs;
__DEVICE__ float abs(float x) { return ::fabsf(x); }
__DEVICE__ double abs(double x) { return ::fabs(x); }
__DEVICE__ float acos(float x) { return ::acosf(x); }
using ::acos;
using ::acosh;
__DEVICE__ float asin(float x) { return ::asinf(x); }
using ::asin;
using ::asinh;
__DEVICE__ float atan(float x) { return ::atanf(x); }
using ::atan;
__DEVICE__ float atan2(float x, float y) { return ::atan2f(x, y); }
using ::atan2;
using ::atanh;
using ::cbrt;
__DEVICE__ float ceil(float x) { return ::ceilf(x); }
using ::ceil;
using ::copysign;
__DEVICE__ float cos(float x) { return ::cosf(x); }
using ::cos;
__DEVICE__ float cosh(float x) { return ::coshf(x); }
using ::cosh;
using ::erf;
using ::erfc;
__DEVICE__ float exp(float x) { return ::expf(x); }
using ::exp;
using ::exp2;
using ::expm1;
__DEVICE__ float fabs(float x) { return ::fabsf(x); }
using ::fabs;
using ::fdim;
__DEVICE__ float floor(float x) { return ::floorf(x); }
using ::floor;
using ::fma;
using ::fmax;
using ::fmin;
__DEVICE__ float fmod(float x, float y) { return ::fmodf(x, y); }
using ::fmod;
__DEVICE__ int fpclassify(float x) {
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL,
                              FP_ZERO, x);
}
__DEVICE__ int fpclassify(double x) {
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL,
                              FP_ZERO, x);
}
__DEVICE__ float frexp(float arg, int *exp) { return ::frexpf(arg, exp); }
using ::frexp;
using ::hypot;
using ::ilogb;
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
using ::labs;
__DEVICE__ float ldexp(float arg, int exp) { return ::ldexpf(arg, exp); }
using ::ldexp;
using ::lgamma;
using ::llabs;
using ::llrint;
__DEVICE__ float log(float x) { return ::logf(x); }
using ::log;
__DEVICE__ float log10(float x) { return ::log10f(x); }
using ::log10;
using ::log1p;
using ::log2;
using ::logb;
using ::lrint;
using ::lround;
__DEVICE__ float modf(float x, float *iptr) { return ::modff(x, iptr); }
using ::modf;
using ::nan;
using ::nanf;
using ::nearbyint;
using ::nextafter;
__DEVICE__ float nexttoward(float from, float to) {
  return __builtin_nexttowardf(from, to);
}
__DEVICE__ double nexttoward(double from, double to) {
  return __builtin_nexttoward(from, to);
}
using ::pow;
__DEVICE__ float pow(float base, float exp) { return ::powf(base, exp); }
__DEVICE__ float pow(float base, int iexp) { return ::powif(base, iexp); }
__DEVICE__ double pow(double base, int iexp) { return ::powi(base, iexp); }
using ::remainder;
using ::remquo;
using ::rint;
using ::round;
using ::scalbln;
using ::scalbn;
__DEVICE__ bool signbit(float x) { return ::__signbitf(x); }
__DEVICE__ bool signbit(double x) { return ::__signbit(x); }
__DEVICE__ float sin(float x) { return ::sinf(x); }
using ::sin;
__DEVICE__ float sinh(float x) { return ::sinhf(x); }
using ::sinh;
__DEVICE__ float sqrt(float x) { return ::sqrtf(x); }
using ::sqrt;
__DEVICE__ float tan(float x) { return ::tanf(x); }
using ::tan;
__DEVICE__ float tanh(float x) { return ::tanhf(x); }
using ::tanh;
using ::tgamma;
using ::trunc;

} // namespace std

#endif
