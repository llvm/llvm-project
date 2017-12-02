// RUN: %clang_cc1 -triple x86_64-unknown-unknown     -w -S -o - -emit-llvm              %s | FileCheck %s --check-prefix=NO__ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-unknown     -w -S -o - -emit-llvm -fmath-errno %s | FileCheck %s --check-prefix=HAS_ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-unknown-gnu -w -S -o - -emit-llvm -fmath-errno %s | FileCheck %s --check-prefix=HAS_ERRNO_GNU
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -w -S -o - -emit-llvm -fmath-errno %s | FileCheck %s --check-prefix=HAS_ERRNO_WIN

// Test attributes and builtin codegen of math library calls.

void foo(double *d, float f, float *fp, long double *l, int *i, const char *c) {
  atan2(f,f);    atan2f(f,f) ;  atan2l(f, f);

// NO__ERRNO: declare double @atan2(double, double) [[READNONE:#[0-9]+]]
// NO__ERRNO: declare float @atan2f(float, float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @atan2l(x86_fp80, x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @atan2(double, double) [[NOT_READNONE:#[0-9]+]]
// HAS_ERRNO: declare float @atan2f(float, float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @atan2l(x86_fp80, x86_fp80) [[NOT_READNONE]]

  copysign(f,f); copysignf(f,f);copysignl(f,f);

// NO__ERRNO: declare double @llvm.copysign.f64(double, double) [[READNONE_INTRINSIC:#[0-9]+]]
// NO__ERRNO: declare float @llvm.copysign.f32(float, float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.copysign.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.copysign.f64(double, double) [[READNONE_INTRINSIC:#[0-9]+]]
// HAS_ERRNO: declare float @llvm.copysign.f32(float, float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.copysign.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]

  fabs(f);       fabsf(f);      fabsl(f);

// NO__ERRNO: declare double @llvm.fabs.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.fabs.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.fabs.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.fabs.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.fabs.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.fabs.f80(x86_fp80) [[READNONE_INTRINSIC]]

  fmod(f,f);     fmodf(f,f);    fmodl(f,f);

// NO__ERRNO: declare double @fmod(double, double) [[READNONE]]
// NO__ERRNO: declare float @fmodf(float, float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @fmodl(x86_fp80, x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @fmod(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @fmodf(float, float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @fmodl(x86_fp80, x86_fp80) [[NOT_READNONE]]

  frexp(f,i);    frexpf(f,i);   frexpl(f,i);

// NO__ERRNO: declare double @frexp(double, i32*) [[NOT_READNONE:#[0-9]+]]
// NO__ERRNO: declare float @frexpf(float, i32*) [[NOT_READNONE]]
// NO__ERRNO: declare x86_fp80 @frexpl(x86_fp80, i32*) [[NOT_READNONE]]
// HAS_ERRNO: declare double @frexp(double, i32*) [[NOT_READNONE]]
// HAS_ERRNO: declare float @frexpf(float, i32*) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @frexpl(x86_fp80, i32*) [[NOT_READNONE]]

  ldexp(f,f);    ldexpf(f,f);   ldexpl(f,f);  

// NO__ERRNO: declare double @ldexp(double, i32) [[READNONE]]
// NO__ERRNO: declare float @ldexpf(float, i32) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @ldexpl(x86_fp80, i32) [[READNONE]]
// HAS_ERRNO: declare double @ldexp(double, i32) [[NOT_READNONE]]
// HAS_ERRNO: declare float @ldexpf(float, i32) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @ldexpl(x86_fp80, i32) [[NOT_READNONE]]

  modf(f,d);       modff(f,fp);      modfl(f,l); 

// NO__ERRNO: declare double @modf(double, double*) [[NOT_READNONE]]
// NO__ERRNO: declare float @modff(float, float*) [[NOT_READNONE]]
// NO__ERRNO: declare x86_fp80 @modfl(x86_fp80, x86_fp80*) [[NOT_READNONE]]
// HAS_ERRNO: declare double @modf(double, double*) [[NOT_READNONE]]
// HAS_ERRNO: declare float @modff(float, float*) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @modfl(x86_fp80, x86_fp80*) [[NOT_READNONE]]

  nan(c);        nanf(c);       nanl(c);  

// NO__ERRNO: declare double @nan(i8*) [[READONLY:#[0-9]+]]
// NO__ERRNO: declare float @nanf(i8*) [[READONLY]]
// NO__ERRNO: declare x86_fp80 @nanl(i8*) [[READONLY]]
// HAS_ERRNO: declare double @nan(i8*) [[READONLY:#[0-9]+]]
// HAS_ERRNO: declare float @nanf(i8*) [[READONLY]]
// HAS_ERRNO: declare x86_fp80 @nanl(i8*) [[READONLY]]

  pow(f,f);        powf(f,f);       powl(f,f);

// NO__ERRNO: declare double @llvm.pow.f64(double, double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.pow.f32(float, float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.pow.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @pow(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @powf(float, float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @powl(x86_fp80, x86_fp80) [[NOT_READNONE]]

  /* math */
  acos(f);       acosf(f);      acosl(f);

// NO__ERRNO: declare double @acos(double) [[READNONE]]
// NO__ERRNO: declare float @acosf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @acosl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @acos(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @acosf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @acosl(x86_fp80) [[NOT_READNONE]]

  acosh(f);      acoshf(f);     acoshl(f);  

// NO__ERRNO: declare double @acosh(double) [[READNONE]]
// NO__ERRNO: declare float @acoshf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @acoshl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @acosh(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @acoshf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @acoshl(x86_fp80) [[NOT_READNONE]]

  asin(f);       asinf(f);      asinl(f); 

// NO__ERRNO: declare double @asin(double) [[READNONE]]
// NO__ERRNO: declare float @asinf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @asinl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @asin(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @asinf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @asinl(x86_fp80) [[NOT_READNONE]]

  asinh(f);      asinhf(f);     asinhl(f);

// NO__ERRNO: declare double @asinh(double) [[READNONE]]
// NO__ERRNO: declare float @asinhf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @asinhl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @asinh(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @asinhf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @asinhl(x86_fp80) [[NOT_READNONE]]

  atan(f);       atanf(f);      atanl(f);

// NO__ERRNO: declare double @atan(double) [[READNONE]]
// NO__ERRNO: declare float @atanf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @atanl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @atan(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @atanf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @atanl(x86_fp80) [[NOT_READNONE]]

  atanh(f);      atanhf(f);     atanhl(f); 

// NO__ERRNO: declare double @atanh(double) [[READNONE]]
// NO__ERRNO: declare float @atanhf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @atanhl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @atanh(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @atanhf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @atanhl(x86_fp80) [[NOT_READNONE]]

  cbrt(f);       cbrtf(f);      cbrtl(f);

// NO__ERRNO: declare double @cbrt(double) [[READNONE]]
// NO__ERRNO: declare float @cbrtf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @cbrtl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @cbrt(double) [[READNONE:#[0-9]+]]
// HAS_ERRNO: declare float @cbrtf(float) [[READNONE]]
// HAS_ERRNO: declare x86_fp80 @cbrtl(x86_fp80) [[READNONE]]

  ceil(f);       ceilf(f);      ceill(f);

// NO__ERRNO: declare double @llvm.ceil.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.ceil.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.ceil.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.ceil.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.ceil.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.ceil.f80(x86_fp80) [[READNONE_INTRINSIC]]

  cos(f);        cosf(f);       cosl(f); 

// NO__ERRNO: declare double @llvm.cos.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.cos.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.cos.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @cos(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @cosf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @cosl(x86_fp80) [[NOT_READNONE]]

  cosh(f);       coshf(f);      coshl(f);

// NO__ERRNO: declare double @cosh(double) [[READNONE]]
// NO__ERRNO: declare float @coshf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @coshl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @cosh(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @coshf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @coshl(x86_fp80) [[NOT_READNONE]]

  erf(f);        erff(f);       erfl(f);

// NO__ERRNO: declare double @erf(double) [[READNONE]]
// NO__ERRNO: declare float @erff(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @erfl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @erf(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @erff(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @erfl(x86_fp80) [[NOT_READNONE]]

  erfc(f);       erfcf(f);      erfcl(f);

// NO__ERRNO: declare double @erfc(double) [[READNONE]]
// NO__ERRNO: declare float @erfcf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @erfcl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @erfc(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @erfcf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @erfcl(x86_fp80) [[NOT_READNONE]]

  exp(f);        expf(f);       expl(f);

// NO__ERRNO: declare double @llvm.exp.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.exp.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.exp.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @exp(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @expf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @expl(x86_fp80) [[NOT_READNONE]]

  exp2(f);       exp2f(f);      exp2l(f); 

// NO__ERRNO: declare double @llvm.exp2.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.exp2.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.exp2.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @exp2(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @exp2f(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @exp2l(x86_fp80) [[NOT_READNONE]]

  expm1(f);      expm1f(f);     expm1l(f);

// NO__ERRNO: declare double @expm1(double) [[READNONE]]
// NO__ERRNO: declare float @expm1f(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @expm1l(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @expm1(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @expm1f(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @expm1l(x86_fp80) [[NOT_READNONE]]

  fdim(f,f);       fdimf(f,f);      fdiml(f,f);

// NO__ERRNO: declare double @fdim(double, double) [[READNONE]]
// NO__ERRNO: declare float @fdimf(float, float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @fdiml(x86_fp80, x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @fdim(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @fdimf(float, float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @fdiml(x86_fp80, x86_fp80) [[NOT_READNONE]]

  floor(f);      floorf(f);     floorl(f);

// NO__ERRNO: declare double @llvm.floor.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.floor.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.floor.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.floor.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.floor.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.floor.f80(x86_fp80) [[READNONE_INTRINSIC]]

  fma(f,f,f);        fmaf(f,f,f);       fmal(f,f,f);

// NO__ERRNO: declare double @llvm.fma.f64(double, double, double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.fma.f32(float, float, float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.fma.f80(x86_fp80, x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @fma(double, double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @fmaf(float, float, float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @fmal(x86_fp80, x86_fp80, x86_fp80) [[NOT_READNONE]]

// On GNU or Win, fma never sets errno, so we can convert to the intrinsic.

// HAS_ERRNO_GNU: declare double @llvm.fma.f64(double, double, double) [[READNONE_INTRINSIC:#[0-9]+]]
// HAS_ERRNO_GNU: declare float @llvm.fma.f32(float, float, float) [[READNONE_INTRINSIC]]
// HAS_ERRNO_GNU: declare x86_fp80 @llvm.fma.f80(x86_fp80, x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]

// HAS_ERRNO_WIN: declare double @llvm.fma.f64(double, double, double) [[READNONE_INTRINSIC:#[0-9]+]]
// HAS_ERRNO_WIN: declare float @llvm.fma.f32(float, float, float) [[READNONE_INTRINSIC]]
// Long double is just double on win, so no f80 use/declaration.
// HAS_ERRNO_WIN-NOT: declare x86_fp80 @llvm.fma.f80(x86_fp80, x86_fp80, x86_fp80)

  fmax(f,f);       fmaxf(f,f);      fmaxl(f,f);

// NO__ERRNO: declare double @llvm.maxnum.f64(double, double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.maxnum.f32(float, float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.maxnum.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.maxnum.f64(double, double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.maxnum.f32(float, float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.maxnum.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]

  fmin(f,f);       fminf(f,f);      fminl(f,f);

// NO__ERRNO: declare double @llvm.minnum.f64(double, double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.minnum.f32(float, float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.minnum.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.minnum.f64(double, double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.minnum.f32(float, float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.minnum.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]

  hypot(f,f);      hypotf(f,f);     hypotl(f,f);

// NO__ERRNO: declare double @hypot(double, double) [[READNONE]]
// NO__ERRNO: declare float @hypotf(float, float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @hypotl(x86_fp80, x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @hypot(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @hypotf(float, float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @hypotl(x86_fp80, x86_fp80) [[NOT_READNONE]]

  ilogb(f);      ilogbf(f);     ilogbl(f); 

// NO__ERRNO: declare i32 @ilogb(double) [[READNONE]]
// NO__ERRNO: declare i32 @ilogbf(float) [[READNONE]]
// NO__ERRNO: declare i32 @ilogbl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare i32 @ilogb(double) [[NOT_READNONE]]
// HAS_ERRNO: declare i32 @ilogbf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare i32 @ilogbl(x86_fp80) [[NOT_READNONE]]

  lgamma(f);     lgammaf(f);    lgammal(f);

// NO__ERRNO: declare double @lgamma(double) [[NOT_READNONE]]
// NO__ERRNO: declare float @lgammaf(float) [[NOT_READNONE]]
// NO__ERRNO: declare x86_fp80 @lgammal(x86_fp80) [[NOT_READNONE]]
// HAS_ERRNO: declare double @lgamma(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @lgammaf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @lgammal(x86_fp80) [[NOT_READNONE]]

  llrint(f);     llrintf(f);    llrintl(f);

// NO__ERRNO: declare i64 @llrint(double) [[READNONE]]
// NO__ERRNO: declare i64 @llrintf(float) [[READNONE]]
// NO__ERRNO: declare i64 @llrintl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare i64 @llrint(double) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @llrintf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @llrintl(x86_fp80) [[NOT_READNONE]]

  llround(f);    llroundf(f);   llroundl(f);

// NO__ERRNO: declare i64 @llround(double) [[READNONE]]
// NO__ERRNO: declare i64 @llroundf(float) [[READNONE]]
// NO__ERRNO: declare i64 @llroundl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare i64 @llround(double) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @llroundf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @llroundl(x86_fp80) [[NOT_READNONE]]

  log(f);        logf(f);       logl(f);

// NO__ERRNO: declare double @llvm.log.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.log.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.log.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @log(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @logf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @logl(x86_fp80) [[NOT_READNONE]]

  log10(f);      log10f(f);     log10l(f);

// NO__ERRNO: declare double @llvm.log10.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.log10.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.log10.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @log10(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @log10f(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @log10l(x86_fp80) [[NOT_READNONE]]

  log1p(f);      log1pf(f);     log1pl(f);

// NO__ERRNO: declare double @log1p(double) [[READNONE]]
// NO__ERRNO: declare float @log1pf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @log1pl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @log1p(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @log1pf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @log1pl(x86_fp80) [[NOT_READNONE]]

  log2(f);       log2f(f);      log2l(f);

// NO__ERRNO: declare double @llvm.log2.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.log2.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.log2.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @log2(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @log2f(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @log2l(x86_fp80) [[NOT_READNONE]]

  logb(f);       logbf(f);      logbl(f);

// NO__ERRNO: declare double @logb(double) [[READNONE]]
// NO__ERRNO: declare float @logbf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @logbl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @logb(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @logbf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @logbl(x86_fp80) [[NOT_READNONE]]

  lrint(f);      lrintf(f);     lrintl(f);

// NO__ERRNO: declare i64 @lrint(double) [[READNONE]]
// NO__ERRNO: declare i64 @lrintf(float) [[READNONE]]
// NO__ERRNO: declare i64 @lrintl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare i64 @lrint(double) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @lrintf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @lrintl(x86_fp80) [[NOT_READNONE]]

  lround(f);     lroundf(f);    lroundl(f);

// NO__ERRNO: declare i64 @lround(double) [[READNONE]]
// NO__ERRNO: declare i64 @lroundf(float) [[READNONE]]
// NO__ERRNO: declare i64 @lroundl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare i64 @lround(double) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @lroundf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @lroundl(x86_fp80) [[NOT_READNONE]]

  nearbyint(f);  nearbyintf(f); nearbyintl(f);

// NO__ERRNO: declare double @llvm.nearbyint.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.nearbyint.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.nearbyint.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.nearbyint.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.nearbyint.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.nearbyint.f80(x86_fp80) [[READNONE_INTRINSIC]]

  nextafter(f,f);  nextafterf(f,f); nextafterl(f,f);

// NO__ERRNO: declare double @nextafter(double, double) [[READNONE]]
// NO__ERRNO: declare float @nextafterf(float, float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @nextafterl(x86_fp80, x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @nextafter(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @nextafterf(float, float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @nextafterl(x86_fp80, x86_fp80) [[NOT_READNONE]]

  nexttoward(f,f); nexttowardf(f,f);nexttowardl(f,f);

// NO__ERRNO: declare double @nexttoward(double, x86_fp80) [[READNONE]]
// NO__ERRNO: declare float @nexttowardf(float, x86_fp80) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @nexttowardl(x86_fp80, x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @nexttoward(double, x86_fp80) [[NOT_READNONE]]
// HAS_ERRNO: declare float @nexttowardf(float, x86_fp80) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @nexttowardl(x86_fp80, x86_fp80) [[NOT_READNONE]]

  remainder(f,f);  remainderf(f,f); remainderl(f,f);

// NO__ERRNO: declare double @remainder(double, double) [[READNONE]]
// NO__ERRNO: declare float @remainderf(float, float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @remainderl(x86_fp80, x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @remainder(double, double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @remainderf(float, float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @remainderl(x86_fp80, x86_fp80) [[NOT_READNONE]]

  remquo(f,f,i);  remquof(f,f,i); remquol(f,f,i);

// NO__ERRNO: declare double @remquo(double, double, i32*) [[NOT_READNONE]]
// NO__ERRNO: declare float @remquof(float, float, i32*) [[NOT_READNONE]]
// NO__ERRNO: declare x86_fp80 @remquol(x86_fp80, x86_fp80, i32*) [[NOT_READNONE]]
// HAS_ERRNO: declare double @remquo(double, double, i32*) [[NOT_READNONE]]
// HAS_ERRNO: declare float @remquof(float, float, i32*) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @remquol(x86_fp80, x86_fp80, i32*) [[NOT_READNONE]]

  rint(f);       rintf(f);      rintl(f);

// NO__ERRNO: declare double @llvm.rint.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.rint.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.rint.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.rint.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.rint.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.rint.f80(x86_fp80) [[READNONE_INTRINSIC]]

  round(f);      roundf(f);     roundl(f);

// NO__ERRNO: declare double @llvm.round.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.round.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.round.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.round.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.round.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.round.f80(x86_fp80) [[READNONE_INTRINSIC]]

  scalbln(f,f);    scalblnf(f,f);   scalblnl(f,f);

// NO__ERRNO: declare double @scalbln(double, i64) [[READNONE]]
// NO__ERRNO: declare float @scalblnf(float, i64) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @scalblnl(x86_fp80, i64) [[READNONE]]
// HAS_ERRNO: declare double @scalbln(double, i64) [[NOT_READNONE]]
// HAS_ERRNO: declare float @scalblnf(float, i64) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @scalblnl(x86_fp80, i64) [[NOT_READNONE]]

  scalbn(f,f);     scalbnf(f,f);    scalbnl(f,f);

// NO__ERRNO: declare double @scalbn(double, i32) [[READNONE]]
// NO__ERRNO: declare float @scalbnf(float, i32) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @scalbnl(x86_fp80, i32) [[READNONE]]
// HAS_ERRNO: declare double @scalbn(double, i32) [[NOT_READNONE]]
// HAS_ERRNO: declare float @scalbnf(float, i32) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @scalbnl(x86_fp80, i32) [[NOT_READNONE]]

  sin(f);        sinf(f);       sinl(f);

// NO__ERRNO: declare double @llvm.sin.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.sin.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.sin.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @sin(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @sinf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @sinl(x86_fp80) [[NOT_READNONE]]

  sinh(f);       sinhf(f);      sinhl(f);

// NO__ERRNO: declare double @sinh(double) [[READNONE]]
// NO__ERRNO: declare float @sinhf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @sinhl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @sinh(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @sinhf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @sinhl(x86_fp80) [[NOT_READNONE]]

  sqrt(f);       sqrtf(f);      sqrtl(f); 

// NO__ERRNO: declare double @llvm.sqrt.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.sqrt.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.sqrt.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @sqrt(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @sqrtf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @sqrtl(x86_fp80) [[NOT_READNONE]]

  tan(f);        tanf(f);       tanl(f);

// NO__ERRNO: declare double @tan(double) [[READNONE]]
// NO__ERRNO: declare float @tanf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @tanl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @tan(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @tanf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @tanl(x86_fp80) [[NOT_READNONE]]

  tanh(f);       tanhf(f);      tanhl(f);

// NO__ERRNO: declare double @tanh(double) [[READNONE]]
// NO__ERRNO: declare float @tanhf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @tanhl(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @tanh(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @tanhf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @tanhl(x86_fp80) [[NOT_READNONE]]

  tgamma(f);     tgammaf(f);    tgammal(f);

// NO__ERRNO: declare double @tgamma(double) [[READNONE]]
// NO__ERRNO: declare float @tgammaf(float) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @tgammal(x86_fp80) [[READNONE]]
// HAS_ERRNO: declare double @tgamma(double) [[NOT_READNONE]]
// HAS_ERRNO: declare float @tgammaf(float) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @tgammal(x86_fp80) [[NOT_READNONE]]

  trunc(f);      truncf(f);     truncl(f);

// NO__ERRNO: declare double @llvm.trunc.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.trunc.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.trunc.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.trunc.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.trunc.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.trunc.f80(x86_fp80) [[READNONE_INTRINSIC]]
};


// NO__ERRNO: attributes [[READNONE]] = { {{.*}}readnone{{.*}} }
// NO__ERRNO: attributes [[READNONE_INTRINSIC]] = { {{.*}}readnone{{.*}} }
// NO__ERRNO: attributes [[NOT_READNONE]] = { nounwind "correctly{{.*}} }
// NO__ERRNO: attributes [[READONLY]] = { {{.*}}readonly{{.*}} }

// HAS_ERRNO: attributes [[NOT_READNONE]] = { nounwind "correctly{{.*}} }
// HAS_ERRNO: attributes [[READNONE_INTRINSIC]] = { {{.*}}readnone{{.*}} }
// HAS_ERRNO: attributes [[READONLY]] = { {{.*}}readonly{{.*}} }
// HAS_ERRNO: attributes [[READNONE]] = { {{.*}}readnone{{.*}} }

// HAS_ERRNO_GNU: attributes [[READNONE_INTRINSIC]] = { {{.*}}readnone{{.*}} }
// HAS_ERRNO_WIN: attributes [[READNONE_INTRINSIC]] = { {{.*}}readnone{{.*}} }
