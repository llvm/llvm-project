#pragma once

// __clang_cuda_(c)math(.h) also provide `abs` which actually belong in
// cstdlib. We could split them out but for now we just include cstdlib from
// cmath.h which is what the systems I've seen do as well.
#include <stdlib.h>

double fabs(double __a);
double acos(double __a);
float acosf(float __a);
double acosh(double __a);
float acoshf(float __a);
double asin(double __a);
float asinf(float __a);
double asinh(double __a);
float asinhf(float __a);
double atan(double __a);
double atan2(double __a, double __b);
float atan2f(float __a, float __b);
float atanf(float __a);
double atanh(double __a);
float atanhf(float __a);
double cbrt(double __a);
float cbrtf(float __a);
double ceil(double __a);
float ceilf(float __a);
double copysign(double __a, double __b);
float copysignf(float __a, float __b);
double cos(double __a);
float cosf(float __a);
double cosh(double __a);
float coshf(float __a);
double cospi(double __a);
float cospif(float __a);
double cyl_bessel_i0(double __a);
float cyl_bessel_i0f(float __a);
double cyl_bessel_i1(double __a);
float cyl_bessel_i1f(float __a);
double erf(double __a);
double erfc(double __a);
float erfcf(float __a);
double erfcinv(double __a);
float erfcinvf(float __a);
double erfcx(double __a);
float erfcxf(float __a);
float erff(float __a);
double erfinv(double __a);
float erfinvf(float __a);
double exp(double __a);
double exp10(double __a);
float exp10f(float __a);
double exp2(double __a);
float exp2f(float __a);
float expf(float __a);
double expm1(double __a);
float expm1f(float __a);
float fabsf(float __a);
double fdim(double __a, double __b);
float fdimf(float __a, float __b);
double fdivide(double __a, double __b);
float fdividef(float __a, float __b);
double floor(double __f);
float floorf(float __f);
double fma(double __a, double __b, double __c);
float fmaf(float __a, float __b, float __c);
double fmax(double __a, double __b);
float fmaxf(float __a, float __b);
double fmin(double __a, double __b);
float fminf(float __a, float __b);
double fmod(double __a, double __b);
float fmodf(float __a, float __b);
double frexp(double __a, int *__b);
float frexpf(float __a, int *__b);
double hypot(double __a, double __b);
float hypotf(float __a, float __b);
int ilogb(double __a);
int ilogbf(float __a);
double j0(double __a);
float j0f(float __a);
double j1(double __a);
float j1f(float __a);
double jn(int __n, double __a);
float jnf(int __n, float __a);
double ldexp(double __a, int __b);
float ldexpf(float __a, int __b);
double lgamma(double __a);
float lgammaf(float __a);
long long llmax(long long __a, long long __b);
long long llmin(long long __a, long long __b);
long long llrint(double __a);
long long llrintf(float __a);
long long llround(double __a);
long long llroundf(float __a);
double log(double __a);
double log10(double __a);
float log10f(float __a);
double log1p(double __a);
float log1pf(float __a);
double log2(double __a);
float log2f(float __a);
double logb(double __a);
float logbf(float __a);
float logf(float __a);
long lrint(double __a);
long lrintf(float __a);
long lround(double __a);
long lroundf(float __a);
int max(int __a, int __b);
int min(int __a, int __b);
double modf(double __a, double *__b);
float modff(float __a, float *__b);
double nearbyint(double __a);
float nearbyintf(float __a);
double nextafter(double __a, double __b);
float nextafterf(float __a, float __b);
double norm(int __dim, const double *__t);
double norm3d(double __a, double __b, double __c);
float norm3df(float __a, float __b, float __c);
double norm4d(double __a, double __b, double __c, double __d);
float norm4df(float __a, float __b, float __c, float __d);
double normcdf(double __a);
float normcdff(float __a);
double normcdfinv(double __a);
float normcdfinvf(float __a);
float normf(int __dim, const float *__t);
double pow(double __a, double __b);
float powf(float __a, float __b);
double powi(double __a, int __b);
float powif(float __a, int __b);
double rcbrt(double __a);
float rcbrtf(float __a);
double remainder(double __a, double __b);
float remainderf(float __a, float __b);
double remquo(double __a, double __b, int *__c);
float remquof(float __a, float __b, int *__c);
double rhypot(double __a, double __b);
float rhypotf(float __a, float __b);
double rint(double __a);
float rintf(float __a);
double rnorm(int __a, const double *__b);
double rnorm3d(double __a, double __b, double __c);
float rnorm3df(float __a, float __b, float __c);
double rnorm4d(double __a, double __b, double __c, double __d);
float rnorm4df(float __a, float __b, float __c, float __d);
float rnormf(int __dim, const float *__t);
double round(double __a);
float roundf(float __a);
double rsqrt(double __a);
float rsqrtf(float __a);
double scalbn(double __a, int __b);
float scalbnf(float __a, int __b);
double scalbln(double __a, long __b);
float scalblnf(float __a, long __b);
double sin(double __a);
void sincos(double __a, double *__s, double *__c);
void sincosf(float __a, float *__s, float *__c);
void sincospi(double __a, double *__s, double *__c);
void sincospif(float __a, float *__s, float *__c);
float sinf(float __a);
double sinh(double __a);
float sinhf(float __a);
double sinpi(double __a);
float sinpif(float __a);
double sqrt(double __a);
float sqrtf(float __a);
double tan(double __a);
float tanf(float __a);
double tanh(double __a);
float tanhf(float __a);
double tgamma(double __a);
float tgammaf(float __a);
double trunc(double __a);
float truncf(float __a);
unsigned long long ullmax(unsigned long long __a,
                          unsigned long long __b);
unsigned long long ullmin(unsigned long long __a,
                          unsigned long long __b);
unsigned int umax(unsigned int __a, unsigned int __b);
unsigned int umin(unsigned int __a, unsigned int __b);
double y0(double __a);
float y0f(float __a);
double y1(double __a);
float y1f(float __a);
double yn(int __a, double __b);
float ynf(int __a, float __b);

/**
 * A positive float constant expression. HUGE_VALF evaluates
 * to +infinity. Used as an error value returned by the built-in
 * math functions.
 */
#define HUGE_VALF (__builtin_huge_valf())

/**
 * A positive double constant expression. HUGE_VAL evaluates
 * to +infinity. Used as an error value returned by the built-in
 * math functions.
 */
#define HUGE_VAL (__builtin_huge_val())

#ifdef __cplusplus
#include <cmath>
#endif
