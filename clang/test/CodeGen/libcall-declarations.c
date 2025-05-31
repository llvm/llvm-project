// RUN: %clang_cc1 -triple x86_64-apple-darwin12 -o - -emit-llvm %s | FileCheck %s -check-prefix=CHECK-NOERRNO
// RUN: %clang_cc1 -triple x86_64 -o - -emit-llvm -fmath-errno %s | FileCheck %s -check-prefix=CHECK-ERRNO
// RUN: %clang_cc1 -triple x86_64-apple-darwin12 -o - -emit-llvm -x c++ %s | FileCheck %s -check-prefix=CHECK-NOERRNO
// RUN: %clang_cc1 -triple x86_64 -o - -emit-llvm -x c++ -fmath-errno %s | FileCheck %s -check-prefix=CHECK-ERRNO

// Prototypes.
#ifdef __cplusplus
extern "C" {
#endif
double atan2(double noundef, double);
float atan2f(float, float);
long double atan2l(long double, long double);
int abs(int);
long int labs(long int);
long long int llabs(long long int);
double copysign(double noundef, double);
float copysignf(float, float);
long double copysignl(long double, long double);
double fabs(double);
float fabsf(float);
long double fabsl(long double);
double fmod(double noundef, double);
float fmodf(float, float);
long double fmodl(long double, long double);
double frexp(double noundef, int *);
float frexpf(float, int *);
long double frexpl(long double noundef, int *);
double ldexp(double noundef, int);
float ldexpf(float, int);
long double ldexpl(long double noundef, int);
double modf(double noundef, double *);
float modff(float, float *);
long double modfl(long double, long double *);
double nan(const char *);
float nanf(const char *);
long double nanl(const char *);
double pow(double noundef, double);
float powf(float, float);
long double powl(long double, long double);
double acos(double);
float acosf(float);
long double acosl(long double);
double acosh(double);
float acoshf(float);
long double acoshl(long double);
double asin(double);
float asinf(float);
long double asinl(long double);
double asinh(double);
float asinhf(float);
long double asinhl(long double);
double atan(double);
float atanf(float);
long double atanl( long double);
double atanh(double);
float atanhf(float);
long double atanhl(long double);
double cbrt(double);
float cbrtf(float);
long double cbrtl(long double);
double ceil(double);
float ceilf(float);
long double ceill(long double);
double cos(double);
float cosf(float);
long double cosl(long double);
double cosh(double);
float coshf(float);
long double coshl(long double);
double erf(double);
float erff(float);
long double erfl(long double);
double erfc(double);
float erfcf(float);
long double erfcl(long double);
double exp(double);
float expf(float);
long double expl(long double);
double exp2(double);
float exp2f(float);
long double exp2l(long double);
double expm1(double);
float expm1f(float);
long double expm1l(long double);
double fdim(double noundef, double);
float fdimf(float, float);
long double fdiml(long double, long double);
double floor(double);
float floorf(float);
long double floorl(long double);
double fma(double noundef, double, double);
float fmaf(float, float, float);
long double fmal(long double, long double, long double);
double fmax(double noundef, double);
float fmaxf(float, float);
long double fmaxl(long double, long double);
double fmin(double noundef, double);
float fminf(float, float);
long double fminl(long double, long double);
double hypot(double noundef, double);
float hypotf(float, float);
long double hypotl(long double, long double);
int ilogb(double);
int ilogbf(float);
int ilogbl(long double);
double lgamma(double);
float lgammaf(float);
long double lgammal(long double);
long long int llrint(double);
long long int llrintf(float);
long long int llrintl(long double);
long long int llround(double);
long long int llroundf(float);
long long int llroundl(long double);
double log(double);
float logf(float);
long double logl(long double);
double log10(double);
float log10f(float);
long double log10l(long double);
double log1p(double);
float log1pf(float);
long double log1pl(long double);
double log2(double);
float log2f(float);
long double log2l(long double);
double logb(double);
float logbf(float);
long double logbl(long double);
long int lrint(double);
long int lrintf(float);
long int lrintl(long double);
long int lround(double);
long int lroundf(float);
long int lroundl(long double);
double nearbyint(double);
float nearbyintf(float);
long double nearbyintl(long double);
double nextafter(double noundef, double);
float nextafterf(float, float);
long double nextafterl(long double, long double);
double nexttoward(double, long double);
float nexttowardf(float, long double);
long double nexttowardl(long double, long double);
double remainder(double noundef, double);
float remainderf(float, float);
long double remainderl(long double, long double);
double rint(double);
float rintf(float);
long double rintl(long double);
double round(double);
float roundf(float);
long double roundl(long double);
double scalbln(double, long int exp);
float scalblnf(float, long int exp);
long double scalblnl(long double, long int exp);
double scalbn(double noundef, int exp);
float scalbnf(float, int exp);
long double scalbnl(long double noundef, int exp);
double sin(double);
float sinf(float);
long double sinl(long double);
double sinh(double);
float sinhf(float);
long double sinhl(long double);
double sqrt(double);
float sqrtf(float);
long double sqrtl(long double);
double tan(double);
float tanf(float);
long double tanl(long double);
double tanh(double);
float tanhf(float);
long double tanhl(long double);
double tgamma(double);
float tgammaf(float);
long double tgammal(long double);
double trunc(double);
float truncf(float);
long double truncl(long double);
double cabs(double _Complex);
float cabsf(float _Complex);
long double cabsl(long double _Complex);
double _Complex cacos(double _Complex);
float _Complex cacosf(float _Complex);
long double _Complex cacosl(long double _Complex);
double _Complex cacosh(double _Complex);
float _Complex cacoshf(float _Complex);
long double _Complex cacoshl(long double _Complex);
double carg(double _Complex);
float cargf(float _Complex);
long double cargl(long double _Complex);
double _Complex casin(double _Complex);
float _Complex casinf(float _Complex);
long double _Complex casinl(long double _Complex);
double _Complex casinh(double _Complex);
float _Complex casinhf(float _Complex);
long double _Complex casinhl(long double _Complex);
double _Complex catan(double _Complex);
float _Complex catanf(float _Complex);
long double _Complex catanl(long double _Complex);
double _Complex catanh(double _Complex);
float _Complex catanhf(float _Complex);
long double _Complex catanhl(long double _Complex);
double _Complex ccos(double _Complex);
float _Complex ccosf(float _Complex);
long double _Complex ccosl(long double _Complex);
double _Complex ccosh(double _Complex);
float _Complex ccoshf(float _Complex);
long double _Complex ccoshl(long double _Complex);
double _Complex cexp(double _Complex);
float _Complex cexpf(float _Complex);
long double _Complex cexpl(long double _Complex);
double cimag(double _Complex);
float cimagf(float _Complex);
long double cimagl(long double _Complex);
double _Complex conj(double _Complex);
float _Complex conjf(float _Complex);
long double _Complex conjl(long double _Complex);
double _Complex clog(double _Complex);
float _Complex clogf(float _Complex);
long double _Complex clogl(long double _Complex);
double _Complex cproj(double _Complex);
float _Complex cprojf(float _Complex);
long double _Complex cprojl(long double _Complex);
double _Complex cpow(double _Complex, _Complex double);
float _Complex cpowf(float _Complex, _Complex float);
long double _Complex cpowl(long double _Complex, _Complex long double);
double creal(double _Complex);
float crealf(float _Complex);
long double creall(long double _Complex);
double _Complex csin(double _Complex);
float _Complex csinf(float _Complex);
long double _Complex csinl(long double _Complex);
double _Complex csinh(double _Complex);
float _Complex csinhf(float _Complex);
long double _Complex csinhl(long double _Complex);
double _Complex csqrt(double _Complex);
float _Complex csqrtf(float _Complex);
long double _Complex csqrtl(long double _Complex);
double _Complex ctan(double _Complex);
float _Complex ctanf(float _Complex);
long double _Complex ctanl(long double _Complex);
double _Complex ctanh(double _Complex);
float _Complex ctanhf(float _Complex);
long double _Complex ctanhl(long double _Complex);

double __sinpi(double);
float __sinpif(float);
double __cospi(double);
float __cospif(float);
double __tanpi(double);
float __tanpif(float);

double __exp10(double);
float __exp10f(float);
#ifdef __cplusplus
}
#endif

// Force emission of the declare statements.
#define F(x) ((void*)x)
void *use[] = {
  F(atan2),      F(atan2f),     F(atan2l),     F(abs),         F(labs),
  F(llabs),      F(copysign),   F(copysignf),  F(copysignl),   F(fabs),
  F(fabsf),      F(fabsl),      F(fmod),       F(fmodf),       F(fmodl),
  F(frexp),      F(frexpf),     F(frexpl),     F(ldexp),       F(ldexpf),
  F(ldexpl),     F(modf),       F(modff),      F(modfl),       F(nan),
  F(nanf),       F(nanl),       F(pow),        F(powf),        F(powl),
  F(acos),       F(acosf),      F(acosl),      F(acosh),       F(acoshf),
  F(acoshl),     F(asin),       F(asinf),      F(asinl),       F(asinh),
  F(asinhf),     F(asinhl),     F(atan),       F(atanf),       F(atanl),
  F(atanh),      F(atanhf),     F(atanhl),     F(cbrt),        F(cbrtf),
  F(cbrtl),      F(ceil),       F(ceilf),      F(ceill),       F(cos),
  F(cosf),       F(cosl),       F(cosh),       F(coshf),       F(coshl),
  F(erf),        F(erff),       F(erfl),       F(erfc),        F(erfcf),
  F(erfcl),      F(exp),        F(expf),       F(expl),        F(exp2),
  F(exp2f),      F(exp2l),      F(expm1),      F(expm1f),      F(expm1l),
  F(fdim),       F(fdimf),      F(fdiml),      F(floor),       F(floorf),
  F(floorl),     F(fma),        F(fmaf),       F(fmal),        F(fmax),
  F(fmaxf),      F(fmaxl),      F(fmin),       F(fminf),       F(fminl),
  F(hypot),      F(hypotf),     F(hypotl),     F(ilogb),       F(ilogbf),
  F(ilogbl),     F(lgamma),     F(lgammaf),    F(lgammal),     F(llrint),
  F(llrintf),    F(llrintl),    F(llround),    F(llroundf),    F(llroundl),
  F(log),        F(logf),       F(logl),       F(log10),       F(log10f),
  F(log10l),     F(log1p),      F(log1pf),     F(log1pl),      F(log2),
  F(log2f),      F(log2l),      F(logb),       F(logbf),       F(logbl),
  F(lrint),      F(lrintf),     F(lrintl),     F(lround),      F(lroundf),
  F(lroundl),    F(nearbyint),  F(nearbyintf), F(nearbyintl),  F(nextafter),
  F(nextafterf), F(nextafterl), F(nexttoward), F(nexttowardf), F(nexttowardl),
  F(remainder),  F(remainderf), F(remainderl), F(rint),        F(rintf),
  F(rintl),      F(round),      F(roundf),     F(roundl),      F(scalbln),
  F(scalblnf),   F(scalblnl),   F(scalbn),     F(scalbnf),     F(scalbnl),
  F(sin),        F(sinf),       F(sinl),       F(sinh),        F(sinhf),
  F(sinhl),      F(sqrt),       F(sqrtf),      F(sqrtl),       F(tan),
  F(tanf),       F(tanl),       F(tanh),       F(tanhf),       F(tanhl),
  F(tgamma),     F(tgammaf),    F(tgammal),    F(trunc),       F(truncf),
  F(truncl),     F(cabs),       F(cabsf),      F(cabsl),       F(cacos),
  F(cacosf),     F(cacosl),     F(cacosh),     F(cacoshf),     F(cacoshl),
  F(carg),       F(cargf),      F(cargl),      F(casin),       F(casinf),
  F(casinl),     F(casinh),     F(casinhf),    F(casinhl),     F(catan),
  F(catanf),     F(catanl),     F(catanh),     F(catanhf),     F(catanhl),
  F(ccos),       F(ccosf),      F(ccosl),      F(ccosh),       F(ccoshf),
  F(ccoshl),     F(cexp),       F(cexpf),      F(cexpl),       F(cimag),
  F(cimagf),     F(cimagl),     F(conj),       F(conjf),       F(conjl),
  F(clog),       F(clogf),      F(clogl),      F(cproj),       F(cprojf),
  F(cprojl),     F(cpow),       F(cpowf),      F(cpowl),       F(creal),
  F(crealf),     F(creall),     F(csin),       F(csinf),       F(csinl),
  F(csinh),      F(csinhf),     F(csinhl),     F(csqrt),       F(csqrtf),
  F(csqrtl),     F(ctan),       F(ctanf),      F(ctanl),       F(ctanh),
  F(ctanhf),     F(ctanhl),     F(__sinpi),    F(__sinpif),    F(__cospi),
  F(__cospif),   F(__tanpi),    F(__tanpif),   F(__exp10),     F(__exp10f)
};

// CHECK-NOERRNO: declare double @atan2(double noundef, double noundef) [[NUWRN:#[0-9]+]]
// CHECK-NOERRNO: declare float @atan2f(float noundef, float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @atan2l(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i32 @abs(i32 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i64 @labs(i64 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i64 @llabs(i64 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @copysign(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @copysignf(float noundef, float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @copysignl(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @fabs(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @fabsf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @fabsl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @fmod(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @fmodf(float noundef, float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @fmodl(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @ldexp(double noundef, i32 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @ldexpf(float noundef, i32 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @ldexpl(x86_fp80 noundef, i32 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @nan(ptr noundef) [[NUWRO:#[0-9]+]]
// CHECK-NOERRNO: declare float @nanf(ptr noundef) [[NUWRO]]
// CHECK-NOERRNO: declare x86_fp80 @nanl(ptr noundef) [[NUWRO]]
// CHECK-NOERRNO: declare double @pow(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @powf(float noundef, float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @powl(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @acos(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @acosf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @acosl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @acosh(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @acoshf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @acoshl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @asin(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @asinf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @asinl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @asinh(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @asinhf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @asinhl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @atan(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @atanf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @atanl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @atanh(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @atanhf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @atanhl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @cbrt(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @cbrtf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @cbrtl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @ceil(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @ceilf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @ceill(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @cos(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @cosf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @cosl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @cosh(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @coshf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @coshl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @erf(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @erff(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @erfl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @erfc(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @erfcf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @erfcl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @exp(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @expf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @expl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @exp2(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @exp2f(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @exp2l(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @expm1(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @expm1f(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @expm1l(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @fdim(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @fdimf(float noundef, float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @fdiml(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @floor(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @floorf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @floorl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @fma(double noundef, double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @fmaf(float noundef, float noundef, float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @fmal(x86_fp80 noundef, x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @fmax(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @fmaxf(float noundef, float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @fmaxl(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @fmin(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @fminf(float noundef, float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @fminl(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @hypot(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @hypotf(float noundef, float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @hypotl(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i32 @ilogb(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i32 @ilogbf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i32 @ilogbl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @lgamma(double noundef) [[NONCONST:#[0-9]+]]
// CHECK-NOERRNO: declare float @lgammaf(float noundef) [[NONCONST]]
// CHECK-NOERRNO: declare x86_fp80 @lgammal(x86_fp80 noundef) [[NONCONST]]
// CHECK-NOERRNO: declare i64 @llrint(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i64 @llrintf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i64 @llrintl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i64 @llround(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i64 @llroundf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i64 @llroundl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @log(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @logf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @logl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @log10(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @log10f(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @log10l(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @log1p(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @log1pf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @log1pl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @log2(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @log2f(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @log2l(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @logb(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @logbf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @logbl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i64 @lrint(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i64 @lrintf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i64 @lrintl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i64 @lround(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i64 @lroundf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare i64 @lroundl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @nearbyint(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @nearbyintf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @nearbyintl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @nextafter(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @nextafterf(float noundef, float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @nextafterl(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @nexttoward(double noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @nexttowardf(float noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @nexttowardl(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @remainder(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @remainderf(float noundef, float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @remainderl(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @rint(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @rintf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @rintl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @round(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @roundf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @roundl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @scalbln(double noundef, i64 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @scalblnf(float noundef, i64 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @scalblnl(x86_fp80 noundef, i64 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @scalbn(double noundef, i32 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @scalbnf(float noundef, i32 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @scalbnl(x86_fp80 noundef, i32 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @sin(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @sinf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @sinl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @sinh(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @sinhf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @sinhl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @sqrt(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @sqrtf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @sqrtl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @tan(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @tanf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @tanl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @tanh(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @tanhf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @tanhl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @tgamma(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @tgammaf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @tgammal(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @trunc(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @truncf(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare x86_fp80 @truncl(x86_fp80 noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @cabs(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @cabsf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @cacos(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @cacosf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @cacosh(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @cacoshf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @carg(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @cargf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @casin(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @casinf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @casinh(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @casinhf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @catan(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @catanf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @catanh(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @catanhf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @ccos(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @ccosf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @ccosh(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @ccoshf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @cexp(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @cexpf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @cimag(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @cimagf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @conj(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @conjf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @clog(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @clogf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @cproj(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @cprojf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @cpow(double noundef, double noundef, double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @cpowf(<2 x float> noundef, <2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @creal(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @crealf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @csin(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @csinf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @csinh(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @csinhf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @csqrt(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @csqrtf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @ctan(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @ctanf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare { double, double } @ctanh(double noundef, double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare <2 x float> @ctanhf(<2 x float> noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @__sinpi(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @__sinpif(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @__cospi(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @__cospif(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @__tanpi(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @__tanpif(float noundef) [[NUWRN]]
// CHECK-NOERRNO: declare double @__exp10(double noundef) [[NUWRN]]
// CHECK-NOERRNO: declare float @__exp10f(float noundef) [[NUWRN]]

// CHECK-ERRNO: declare i32 @abs(i32 noundef) [[NUWRN:#[0-9]+]]
// CHECK-ERRNO: declare i64 @labs(i64 noundef) [[NUWRN]]
// CHECK-ERRNO: declare i64 @llabs(i64 noundef) [[NUWRN]]
// CHECK-ERRNO: declare double @copysign(double noundef, double noundef) [[NUWRN]]
// CHECK-ERRNO: declare float @copysignf(float noundef, float noundef) [[NUWRN]]
// CHECK-ERRNO: declare x86_fp80 @copysignl(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-ERRNO: declare double @fabs(double noundef) [[NUWRN]]
// CHECK-ERRNO: declare float @fabsf(float noundef) [[NUWRN]]
// CHECK-ERRNO: declare x86_fp80 @fabsl(x86_fp80 noundef) [[NUWRN]]
// CHECK-ERRNO: declare double @nan(ptr noundef) [[NUWRO:#[0-9]+]]
// CHECK-ERRNO: declare float @nanf(ptr noundef) [[NUWRO]]
// CHECK-ERRNO: declare x86_fp80 @nanl(ptr noundef) [[NUWRO]]
// CHECK-ERRNO: declare double @ceil(double noundef) [[NUWRN]]
// CHECK-ERRNO: declare float @ceilf(float noundef) [[NUWRN]]
// CHECK-ERRNO: declare x86_fp80 @ceill(x86_fp80 noundef) [[NUWRN]]
// CHECK-ERRNO: declare double @floor(double noundef) [[NUWRN]]
// CHECK-ERRNO: declare float @floorf(float noundef) [[NUWRN]]
// CHECK-ERRNO: declare x86_fp80 @floorl(x86_fp80 noundef) [[NUWRN]]
// CHECK-ERRNO: declare double @fmax(double noundef, double noundef) [[NUWRN]]
// CHECK-ERRNO: declare float @fmaxf(float noundef, float noundef) [[NUWRN]]
// CHECK-ERRNO: declare x86_fp80 @fmaxl(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-ERRNO: declare double @fmin(double noundef, double noundef) [[NUWRN]]
// CHECK-ERRNO: declare float @fminf(float noundef, float noundef) [[NUWRN]]
// CHECK-ERRNO: declare x86_fp80 @fminl(x86_fp80 noundef, x86_fp80 noundef) [[NUWRN]]
// CHECK-ERRNO: declare double @lgamma(double noundef) [[NONCONST:#[0-9]+]]
// CHECK-ERRNO: declare float @lgammaf(float noundef) [[NONCONST]]
// CHECK-ERRNO: declare x86_fp80 @lgammal(x86_fp80 noundef) [[NONCONST]]
// CHECK-ERRNO: declare double @nearbyint(double noundef) [[NUWRN]]
// CHECK-ERRNO: declare float @nearbyintf(float noundef) [[NUWRN]]
// CHECK-ERRNO: declare x86_fp80 @nearbyintl(x86_fp80 noundef) [[NUWRN]]
// CHECK-ERRNO: declare double @rint(double noundef) [[NUWRN]]
// CHECK-ERRNO: declare float @rintf(float noundef) [[NUWRN]]
// CHECK-ERRNO: declare x86_fp80 @rintl(x86_fp80 noundef) [[NUWRN]]
// CHECK-ERRNO: declare double @round(double noundef) [[NUWRN]]
// CHECK-ERRNO: declare float @roundf(float noundef) [[NUWRN]]
// CHECK-ERRNO: declare x86_fp80 @roundl(x86_fp80 noundef) [[NUWRN]]
// CHECK-ERRNO: declare double @trunc(double noundef) [[NUWRN]]
// CHECK-ERRNO: declare float @truncf(float noundef) [[NUWRN]]
// CHECK-ERRNO: declare x86_fp80 @truncl(x86_fp80 noundef) [[NUWRN]]
// CHECK-ERRNO: declare double @cabs(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare float @cabsf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @cacos(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @cacosf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @cacosh(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @cacoshf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare double @carg(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare float @cargf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @casin(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @casinf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @casinh(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @casinhf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @catan(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @catanf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @catanh(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @catanhf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @ccos(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @ccosf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @ccosh(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @ccoshf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @cexp(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @cexpf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare double @cimag(double noundef, double noundef) [[NUWRN]]
// CHECK-ERRNO: declare float @cimagf(<2 x float> noundef) [[NUWRN]]
// CHECK-ERRNO: declare { double, double } @conj(double noundef, double noundef) [[NUWRN]]
// CHECK-ERRNO: declare <2 x float> @conjf(<2 x float> noundef) [[NUWRN]]
// CHECK-ERRNO: declare { double, double } @clog(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @clogf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @cproj(double noundef, double noundef) [[NUWRN]]
// CHECK-ERRNO: declare <2 x float> @cprojf(<2 x float> noundef) [[NUWRN]]
// CHECK-ERRNO: declare { double, double } @cpow(double noundef, double noundef, double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @cpowf(<2 x float> noundef, <2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare double @creal(double noundef, double noundef) [[NUWRN]]
// CHECK-ERRNO: declare float @crealf(<2 x float> noundef) [[NUWRN]]
// CHECK-ERRNO: declare { double, double } @csin(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @csinf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @csinh(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @csinhf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @csqrt(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @csqrtf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @ctan(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @ctanf(<2 x float> noundef) [[NONCONST]]
// CHECK-ERRNO: declare { double, double } @ctanh(double noundef, double noundef) [[NONCONST]]
// CHECK-ERRNO: declare <2 x float> @ctanhf(<2 x float> noundef) [[NONCONST]]

// CHECK-NOERRNO: attributes [[NUWRN]] = { nounwind willreturn memory(none){{.*}} }
// CHECK-NOERRNO: attributes [[NUWRO]] = { nounwind willreturn memory(read){{.*}} }

// CHECK-ERRNO: attributes [[NUWRN]] = { nounwind willreturn memory(none){{.*}} }
// CHECK-ERRNO: attributes [[NUWRO]] = { nounwind willreturn memory(read){{.*}} }
