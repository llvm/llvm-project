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

// CUDA lets us use various std math functions on the device side.  This file
// works in concert with __clang_cuda_math_forward_declares.h to make this work.
//
// Specifically, the forward-declares header declares __device__ overloads for
// these functions in the global namespace, then pulls them into namespace std
// with 'using' statements.  Then this file implements those functions, after
// the implementations have been pulled in.
//
// It's important that we declare the functions in the global namespace and pull
// them into namespace std with using statements, as opposed to simply declaring
// these functions in namespace std, because our device functions need to
// overload the standard library functions, which may be declared in the global
// namespace or in std, depending on the degree of conformance of the stdlib
// implementation.  Declaring in the global namespace and pulling into namespace
// std covers all of the known knowns.

#define __DEVICE__ static __device__ __inline__ __attribute__((always_inline))

__DEVICE__ long long abs(long long __n) { return ::llabs(__n); }
__DEVICE__ long abs(long __n) { return ::labs(__n); }
__DEVICE__ float abs(float __x) { return ::fabsf(__x); }
__DEVICE__ double abs(double __x) { return ::fabs(__x); }
__DEVICE__ float acos(float __x) { return ::acosf(__x); }
__DEVICE__ float asin(float __x) { return ::asinf(__x); }
__DEVICE__ float atan(float __x) { return ::atanf(__x); }
__DEVICE__ float atan2(float __x, float __y) { return ::atan2f(__x, __y); }
__DEVICE__ float ceil(float __x) { return ::ceilf(__x); }
__DEVICE__ float cos(float __x) { return ::cosf(__x); }
__DEVICE__ float cosh(float __x) { return ::coshf(__x); }
__DEVICE__ float exp(float __x) { return ::expf(__x); }
__DEVICE__ float fabs(float __x) { return ::fabsf(__x); }
__DEVICE__ float floor(float __x) { return ::floorf(__x); }
__DEVICE__ float fmod(float __x, float __y) { return ::fmodf(__x, __y); }
__DEVICE__ int fpclassify(float __x) {
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL,
                              FP_ZERO, __x);
}
__DEVICE__ int fpclassify(double __x) {
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL,
                              FP_ZERO, __x);
}
__DEVICE__ float frexp(float __arg, int *__exp) {
  return ::frexpf(__arg, __exp);
}
__DEVICE__ bool isinf(float __x) { return ::__isinff(__x); }
__DEVICE__ bool isinf(double __x) { return ::__isinf(__x); }
__DEVICE__ bool isfinite(float __x) { return ::__finitef(__x); }
__DEVICE__ bool isfinite(double __x) { return ::__finite(__x); }
__DEVICE__ bool isgreater(float __x, float __y) {
  return __builtin_isgreater(__x, __y);
}
__DEVICE__ bool isgreater(double __x, double __y) {
  return __builtin_isgreater(__x, __y);
}
__DEVICE__ bool isgreaterequal(float __x, float __y) {
  return __builtin_isgreaterequal(__x, __y);
}
__DEVICE__ bool isgreaterequal(double __x, double __y) {
  return __builtin_isgreaterequal(__x, __y);
}
__DEVICE__ bool isless(float __x, float __y) {
  return __builtin_isless(__x, __y);
}
__DEVICE__ bool isless(double __x, double __y) {
  return __builtin_isless(__x, __y);
}
__DEVICE__ bool islessequal(float __x, float __y) {
  return __builtin_islessequal(__x, __y);
}
__DEVICE__ bool islessequal(double __x, double __y) {
  return __builtin_islessequal(__x, __y);
}
__DEVICE__ bool islessgreater(float __x, float __y) {
  return __builtin_islessgreater(__x, __y);
}
__DEVICE__ bool islessgreater(double __x, double __y) {
  return __builtin_islessgreater(__x, __y);
}
__DEVICE__ bool isnan(float __x) { return ::__isnanf(__x); }
__DEVICE__ bool isnan(double __x) { return ::__isnan(__x); }
__DEVICE__ bool isnormal(float __x) { return __builtin_isnormal(__x); }
__DEVICE__ bool isnormal(double __x) { return __builtin_isnormal(__x); }
__DEVICE__ bool isunordered(float __x, float __y) {
  return __builtin_isunordered(__x, __y);
}
__DEVICE__ bool isunordered(double __x, double __y) {
  return __builtin_isunordered(__x, __y);
}
__DEVICE__ float ldexp(float __arg, int __exp) {
  return ::ldexpf(__arg, __exp);
}
__DEVICE__ float log(float __x) { return ::logf(__x); }
__DEVICE__ float log10(float __x) { return ::log10f(__x); }
__DEVICE__ float modf(float __x, float *__iptr) { return ::modff(__x, __iptr); }
__DEVICE__ float nexttoward(float __from, float __to) {
  return __builtin_nexttowardf(__from, __to);
}
__DEVICE__ double nexttoward(double __from, double __to) {
  return __builtin_nexttoward(__from, __to);
}
__DEVICE__ float pow(float __base, float __exp) {
  return ::powf(__base, __exp);
}
__DEVICE__ float pow(float __base, int __iexp) {
  return ::powif(__base, __iexp);
}
__DEVICE__ double pow(double __base, int __iexp) {
  return ::powi(__base, __iexp);
}
__DEVICE__ bool signbit(float __x) { return ::__signbitf(__x); }
__DEVICE__ bool signbit(double __x) { return ::__signbit(__x); }
__DEVICE__ float sin(float __x) { return ::sinf(__x); }
__DEVICE__ float sinh(float __x) { return ::sinhf(__x); }
__DEVICE__ float sqrt(float __x) { return ::sqrtf(__x); }
__DEVICE__ float tan(float __x) { return ::tanf(__x); }
__DEVICE__ float tanh(float __x) { return ::tanhf(__x); }

#undef __DEVICE__

#endif
