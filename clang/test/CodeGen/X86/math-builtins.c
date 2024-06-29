// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -o - -emit-llvm %s | FileCheck %s -check-prefix=NO__ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -o - -emit-llvm -fmath-errno %s | FileCheck %s -check-prefix=HAS_ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -o - -emit-llvm -disable-llvm-passes -O2 %s | FileCheck %s -check-prefix=NO__ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -o - -emit-llvm -disable-llvm-passes -O2 -fmath-errno %s | FileCheck %s -check-prefix=HAS_ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-unknown-gnu -w -o - -emit-llvm -fmath-errno %s | FileCheck %s --check-prefix=HAS_ERRNO_GNU
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -w -o - -emit-llvm -fmath-errno %s | FileCheck %s --check-prefix=HAS_ERRNO_WIN

// Test attributes and codegen of math builtins.

void foo(double *d, float f, float *fp, long double *l, int *i, const char *c) {
  f = __builtin_fmod(f,f);    f = __builtin_fmodf(f,f);   f =  __builtin_fmodl(f,f);  f = __builtin_fmodf128(f,f);

// NO__ERRNO: frem double
// NO__ERRNO: frem float
// NO__ERRNO: frem x86_fp80
// NO__ERRNO: frem fp128

// NO__ERRNO: [[FREXP_F64:%.+]] = call { double, i32 } @llvm.frexp.f64.i32(double %{{.+}})
// NO__ERRNO-NEXT: [[FREXP_F64_1:%.+]] = extractvalue { double, i32 } [[FREXP_F64]], 1
// NO__ERRNO-NEXT: store i32 [[FREXP_F64_1]], ptr %{{.+}}, align 4
// NO__ERRNO-NEXT: [[FREXP_F64_0:%.+]] = extractvalue { double, i32 } [[FREXP_F64]], 0

// NO__ERRNO: [[FREXP_F32:%.+]] = call { float, i32 } @llvm.frexp.f32.i32(float %{{.+}})
// NO__ERRNO-NEXT: [[FREXP_F32_1:%.+]] = extractvalue { float, i32 } [[FREXP_F32]], 1
// NO__ERRNO-NEXT: store i32 [[FREXP_F32_1]], ptr %{{.+}}, align 4
// NO__ERRNO-NEXT: [[FREXP_F32_0:%.+]] = extractvalue { float, i32 } [[FREXP_F32]], 0


// NO__ERRNO: [[FREXP_F80:%.+]] = call { x86_fp80, i32 } @llvm.frexp.f80.i32(x86_fp80 %{{.+}})
// NO__ERRNO-NEXT: [[FREXP_F80_1:%.+]] = extractvalue { x86_fp80, i32 } [[FREXP_F80]], 1
// NO__ERRNO-NEXT: store i32 [[FREXP_F80_1]], ptr %{{.+}}, align 4
// NO__ERRNO-NEXT: [[FREXP_F80_0:%.+]] = extractvalue { x86_fp80, i32 } [[FREXP_F80]], 0


// NO__ERRNO: [[FREXP_F128:%.+]] = call { fp128, i32 } @llvm.frexp.f128.i32(fp128 %{{.+}})
// NO__ERRNO-NEXT: [[FREXP_F128_1:%.+]] = extractvalue { fp128, i32 } [[FREXP_F128]], 1
// NO__ERRNO-NEXT: store i32 [[FREXP_F128_1]], ptr %{{.+}}, align 4
// NO__ERRNO-NEXT: [[FREXP_F128_0:%.+]] = extractvalue { fp128, i32 } [[FREXP_F128]], 0


// HAS_ERRNO: declare double @fmod(double noundef, double noundef) [[NOT_READNONE:#[0-9]+]]
// HAS_ERRNO: declare float @fmodf(float noundef, float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @fmodl(x86_fp80 noundef, x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @fmodf128(fp128 noundef, fp128 noundef) [[NOT_READNONE]]

  __builtin_atan2(f,f);    __builtin_atan2f(f,f) ;  __builtin_atan2l(f, f); __builtin_atan2f128(f,f);

// NO__ERRNO: declare double @atan2(double noundef, double noundef) [[READNONE:#[0-9]+]]
// NO__ERRNO: declare float @atan2f(float noundef, float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @atan2l(x86_fp80 noundef, x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @atan2f128(fp128 noundef, fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @atan2(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @atan2f(float noundef, float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @atan2l(x86_fp80 noundef, x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @atan2f128(fp128 noundef, fp128 noundef) [[NOT_READNONE]]

  __builtin_copysign(f,f); __builtin_copysignf(f,f); __builtin_copysignl(f,f); __builtin_copysignf128(f,f);

// NO__ERRNO: declare double @llvm.copysign.f64(double, double) [[READNONE_INTRINSIC:#[0-9]+]]
// NO__ERRNO: declare float @llvm.copysign.f32(float, float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.copysign.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.copysign.f128(fp128, fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.copysign.f64(double, double) [[READNONE_INTRINSIC:#[0-9]+]]
// HAS_ERRNO: declare float @llvm.copysign.f32(float, float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.copysign.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare fp128 @llvm.copysign.f128(fp128, fp128) [[READNONE_INTRINSIC]]

  __builtin_fabs(f);       __builtin_fabsf(f);      __builtin_fabsl(f); __builtin_fabsf128(f);

// NO__ERRNO: declare double @llvm.fabs.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.fabs.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.fabs.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.fabs.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.fabs.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.fabs.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.fabs.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare fp128 @llvm.fabs.f128(fp128) [[READNONE_INTRINSIC]]

  __builtin_frexp(f,i);    __builtin_frexpf(f,i);   __builtin_frexpl(f,i); __builtin_frexpf128(f,i);

// NO__ERRNO: declare { double, i32 } @llvm.frexp.f64.i32(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare { float, i32 } @llvm.frexp.f32.i32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare { x86_fp80, i32 } @llvm.frexp.f80.i32(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare { fp128, i32 } @llvm.frexp.f128.i32(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare { double, i32 } @llvm.frexp.f64.i32(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare { float, i32 } @llvm.frexp.f32.i32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare { x86_fp80, i32 } @llvm.frexp.f80.i32(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare { fp128, i32 } @llvm.frexp.f128.i32(fp128) [[READNONE_INTRINSIC]]

  __builtin_huge_val();    __builtin_huge_valf();   __builtin_huge_vall(); __builtin_huge_valf128();

// NO__ERRNO-NOT: .huge
// NO__ERRNO-NOT: @huge
// HAS_ERRNO-NOT: .huge
// HAS_ERRNO-NOT: @huge

  __builtin_inf();    __builtin_inff();   __builtin_infl(); __builtin_inff128();

// NO__ERRNO-NOT: .inf
// NO__ERRNO-NOT: @inf
// HAS_ERRNO-NOT: .inf
// HAS_ERRNO-NOT: @inf

  __builtin_ldexp(f,f);    __builtin_ldexpf(f,f);   __builtin_ldexpl(f,f);  __builtin_ldexpf128(f,f);

// NO__ERRNO: declare double @llvm.ldexp.f64.i32(double, i32) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.ldexp.f32.i32(float, i32) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.ldexp.f80.i32(x86_fp80, i32) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.ldexp.f128.i32(fp128, i32) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @ldexp(double noundef, i32 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @ldexpf(float noundef, i32 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @ldexpl(x86_fp80 noundef, i32 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @ldexpf128(fp128 noundef, i32 noundef) [[NOT_READNONE]]

  __builtin_modf(f,d);       __builtin_modff(f,fp);      __builtin_modfl(f,l); __builtin_modff128(f,l);

// NO__ERRNO: declare double @modf(double noundef, ptr noundef) [[NOT_READNONE:#[0-9]+]]
// NO__ERRNO: declare float @modff(float noundef, ptr noundef) [[NOT_READNONE]]
// NO__ERRNO: declare x86_fp80 @modfl(x86_fp80 noundef, ptr noundef) [[NOT_READNONE]]
// NO__ERRNO: declare fp128 @modff128(fp128 noundef, ptr noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare double @modf(double noundef, ptr noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @modff(float noundef, ptr noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @modfl(x86_fp80 noundef, ptr noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @modff128(fp128 noundef, ptr noundef) [[NOT_READNONE]]

  __builtin_nan(c);        __builtin_nanf(c);       __builtin_nanl(c); __builtin_nanf128(c);

// NO__ERRNO: declare double @nan(ptr noundef) [[PURE:#[0-9]+]]
// NO__ERRNO: declare float @nanf(ptr noundef) [[PURE]]
// NO__ERRNO: declare x86_fp80 @nanl(ptr noundef) [[PURE]]
// NO__ERRNO: declare fp128 @nanf128(ptr noundef) [[PURE]]
// HAS_ERRNO: declare double @nan(ptr noundef) [[PURE:#[0-9]+]]
// HAS_ERRNO: declare float @nanf(ptr noundef) [[PURE]]
// HAS_ERRNO: declare x86_fp80 @nanl(ptr noundef) [[PURE]]
// HAS_ERRNO: declare fp128 @nanf128(ptr noundef) [[PURE]]

  __builtin_nans(c);        __builtin_nansf(c);       __builtin_nansl(c); __builtin_nansf128(c);

// NO__ERRNO: declare double @nans(ptr noundef) [[PURE]]
// NO__ERRNO: declare float @nansf(ptr noundef) [[PURE]]
// NO__ERRNO: declare x86_fp80 @nansl(ptr noundef) [[PURE]]
// NO__ERRNO: declare fp128 @nansf128(ptr noundef) [[PURE]]
// HAS_ERRNO: declare double @nans(ptr noundef) [[PURE]]
// HAS_ERRNO: declare float @nansf(ptr noundef) [[PURE]]
// HAS_ERRNO: declare x86_fp80 @nansl(ptr noundef) [[PURE]]
// HAS_ERRNO: declare fp128 @nansf128(ptr noundef) [[PURE]]

  __builtin_pow(f,f);        __builtin_powf(f,f);       __builtin_powl(f,f); __builtin_powf128(f,f);

// NO__ERRNO: declare double @llvm.pow.f64(double, double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.pow.f32(float, float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.pow.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.pow.f128(fp128, fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @pow(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @powf(float noundef, float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @powl(x86_fp80 noundef, x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @powf128(fp128 noundef, fp128 noundef) [[NOT_READNONE]]

  __builtin_powi(f,f);        __builtin_powif(f,f);       __builtin_powil(f,f);

// NO__ERRNO: declare double @llvm.powi.f64.i32(double, i32) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.powi.f32.i32(float, i32) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.powi.f80.i32(x86_fp80, i32) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.powi.f64.i32(double, i32) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.powi.f32.i32(float, i32) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.powi.f80.i32(x86_fp80, i32) [[READNONE_INTRINSIC]]

  /* math */
  __builtin_acos(f);       __builtin_acosf(f);      __builtin_acosl(f); __builtin_acosf128(f);

// NO__ERRNO: declare double @acos(double noundef) [[READNONE]]
// NO__ERRNO: declare float @acosf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @acosl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @acosf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @acos(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @acosf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @acosl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @acosf128(fp128 noundef) [[NOT_READNONE]]

  __builtin_acosh(f);      __builtin_acoshf(f);     __builtin_acoshl(f);  __builtin_acoshf128(f);

// NO__ERRNO: declare double @acosh(double noundef) [[READNONE]]
// NO__ERRNO: declare float @acoshf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @acoshl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @acoshf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @acosh(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @acoshf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @acoshl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @acoshf128(fp128 noundef) [[NOT_READNONE]]

  __builtin_asin(f);       __builtin_asinf(f);      __builtin_asinl(f); __builtin_asinf128(f);

// NO__ERRNO: declare double @asin(double noundef) [[READNONE]]
// NO__ERRNO: declare float @asinf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @asinl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @asinf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @asin(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @asinf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @asinl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @asinf128(fp128 noundef) [[NOT_READNONE]]

  __builtin_asinh(f);      __builtin_asinhf(f);     __builtin_asinhl(f); __builtin_asinhf128(f);

// NO__ERRNO: declare double @asinh(double noundef) [[READNONE]]
// NO__ERRNO: declare float @asinhf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @asinhl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @asinhf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @asinh(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @asinhf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @asinhl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @asinhf128(fp128 noundef) [[NOT_READNONE]]

  __builtin_atan(f);       __builtin_atanf(f);      __builtin_atanl(f); __builtin_atanf128(f);

// NO__ERRNO: declare double @atan(double noundef) [[READNONE]]
// NO__ERRNO: declare float @atanf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @atanl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @atanf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @atan(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @atanf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @atanl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @atanf128(fp128 noundef) [[NOT_READNONE]]

  __builtin_atanh(f);      __builtin_atanhf(f);     __builtin_atanhl(f); __builtin_atanhf128(f);

// NO__ERRNO: declare double @atanh(double noundef) [[READNONE]]
// NO__ERRNO: declare float @atanhf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @atanhl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @atanhf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @atanh(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @atanhf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @atanhl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @atanhf128(fp128 noundef) [[NOT_READNONE]]

  __builtin_cbrt(f);       __builtin_cbrtf(f);      __builtin_cbrtl(f); __builtin_cbrtf128(f);

// NO__ERRNO: declare double @cbrt(double noundef) [[READNONE]]
// NO__ERRNO: declare float @cbrtf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @cbrtl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @cbrtf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @cbrt(double noundef) [[READNONE:#[0-9]+]]
// HAS_ERRNO: declare float @cbrtf(float noundef) [[READNONE]]
// HAS_ERRNO: declare x86_fp80 @cbrtl(x86_fp80 noundef) [[READNONE]]
// HAS_ERRNO: declare fp128 @cbrtf128(fp128 noundef) [[READNONE]]

  __builtin_ceil(f);       __builtin_ceilf(f);      __builtin_ceill(f); __builtin_ceilf128(f);

// NO__ERRNO: declare double @llvm.ceil.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.ceil.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.ceil.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.ceil.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.ceil.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.ceil.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.ceil.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare fp128 @llvm.ceil.f128(fp128) [[READNONE_INTRINSIC]]

  __builtin_cos(f);        __builtin_cosf(f);       __builtin_cosl(f); __builtin_cosf128(f);

// NO__ERRNO: declare double @llvm.cos.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.cos.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.cos.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.cos.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @cos(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @cosf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @cosl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @cosf128(fp128 noundef) [[NOT_READNONE]]

  __builtin_cosh(f);       __builtin_coshf(f);      __builtin_coshl(f); __builtin_coshf128(f);

// NO__ERRNO: declare double @cosh(double noundef) [[READNONE]]
// NO__ERRNO: declare float @coshf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @coshl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @coshf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @cosh(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @coshf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @coshl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @coshf128(fp128 noundef) [[NOT_READNONE]]

  __builtin_erf(f);        __builtin_erff(f);       __builtin_erfl(f); __builtin_erff128(f);

// NO__ERRNO: declare double @erf(double noundef) [[READNONE]]
// NO__ERRNO: declare float @erff(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @erfl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @erff128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @erf(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @erff(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @erfl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @erff128(fp128 noundef) [[NOT_READNONE]]

__builtin_erfc(f);       __builtin_erfcf(f);      __builtin_erfcl(f); __builtin_erfcf128(f);

// NO__ERRNO: declare double @erfc(double noundef) [[READNONE]]
// NO__ERRNO: declare float @erfcf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @erfcl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @erfcf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @erfc(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @erfcf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @erfcl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @erfcf128(fp128 noundef) [[NOT_READNONE]]

__builtin_exp(f);        __builtin_expf(f);       __builtin_expl(f); __builtin_expf128(f);

// NO__ERRNO: declare double @llvm.exp.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.exp.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.exp.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.exp.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @exp(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @expf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @expl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @expf128(fp128 noundef) [[NOT_READNONE]]

__builtin_exp2(f);       __builtin_exp2f(f);      __builtin_exp2l(f); __builtin_exp2f128(f);

// NO__ERRNO: declare double @llvm.exp2.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.exp2.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.exp2.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.exp2.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @exp2(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @exp2f(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @exp2l(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @exp2f128(fp128 noundef) [[NOT_READNONE]]

__builtin_exp10(f);       __builtin_exp10f(f);      __builtin_exp10l(f); __builtin_exp10f128(f);

// NO__ERRNO: declare double @llvm.exp10.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.exp10.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.exp10.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.exp10.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @exp10(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @exp10f(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @exp10l(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @exp10f128(fp128 noundef) [[NOT_READNONE]]

__builtin_expm1(f);      __builtin_expm1f(f);     __builtin_expm1l(f); __builtin_expm1f128(f);

// NO__ERRNO: declare double @expm1(double noundef) [[READNONE]]
// NO__ERRNO: declare float @expm1f(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @expm1l(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @expm1f128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @expm1(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @expm1f(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @expm1l(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @expm1f128(fp128 noundef) [[NOT_READNONE]]

__builtin_fdim(f,f);       __builtin_fdimf(f,f);      __builtin_fdiml(f,f); __builtin_fdimf128(f,f);

// NO__ERRNO: declare double @fdim(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare float @fdimf(float noundef, float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @fdiml(x86_fp80 noundef, x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @fdimf128(fp128 noundef, fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @fdim(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @fdimf(float noundef, float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @fdiml(x86_fp80 noundef, x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @fdimf128(fp128 noundef, fp128 noundef) [[NOT_READNONE]]

__builtin_floor(f);      __builtin_floorf(f);     __builtin_floorl(f); __builtin_floorf128(f);

// NO__ERRNO: declare double @llvm.floor.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.floor.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.floor.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.floor.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.floor.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.floor.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.floor.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare fp128 @llvm.floor.f128(fp128) [[READNONE_INTRINSIC]]

__builtin_fma(f,f,f);        __builtin_fmaf(f,f,f);       __builtin_fmal(f,f,f); __builtin_fmaf128(f,f,f);

// NO__ERRNO: declare double @llvm.fma.f64(double, double, double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.fma.f32(float, float, float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.fma.f80(x86_fp80, x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.fma.f128(fp128, fp128, fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @fma(double noundef, double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @fmaf(float noundef, float noundef, float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @fmal(x86_fp80 noundef, x86_fp80 noundef, x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @fmaf128(fp128 noundef, fp128 noundef, fp128 noundef) [[NOT_READNONE]]

// On GNU or Win, fma never sets errno, so we can convert to the intrinsic.

// HAS_ERRNO_GNU: declare double @llvm.fma.f64(double, double, double) [[READNONE_INTRINSIC:#[0-9]+]]
// HAS_ERRNO_GNU: declare float @llvm.fma.f32(float, float, float) [[READNONE_INTRINSIC]]
// HAS_ERRNO_GNU: declare x86_fp80 @llvm.fma.f80(x86_fp80, x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]

// HAS_ERRNO_WIN: declare double @llvm.fma.f64(double, double, double) [[READNONE_INTRINSIC:#[0-9]+]]
// HAS_ERRNO_WIN: declare float @llvm.fma.f32(float, float, float) [[READNONE_INTRINSIC]]
// Long double is just double on win, so no f80 use/declaration.
// HAS_ERRNO_WIN-NOT: declare x86_fp80 @llvm.fma.f80(x86_fp80, x86_fp80, x86_fp80)

__builtin_fmax(f,f);       __builtin_fmaxf(f,f);      __builtin_fmaxl(f,f); __builtin_fmaxf128(f,f);

// NO__ERRNO: declare double @llvm.maxnum.f64(double, double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.maxnum.f32(float, float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.maxnum.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.maxnum.f128(fp128, fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.maxnum.f64(double, double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.maxnum.f32(float, float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.maxnum.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare fp128 @llvm.maxnum.f128(fp128, fp128) [[READNONE_INTRINSIC]]

__builtin_fmin(f,f);       __builtin_fminf(f,f);      __builtin_fminl(f,f); __builtin_fminf128(f,f);

// NO__ERRNO: declare double @llvm.minnum.f64(double, double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.minnum.f32(float, float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.minnum.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.minnum.f128(fp128, fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.minnum.f64(double, double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.minnum.f32(float, float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.minnum.f80(x86_fp80, x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare fp128 @llvm.minnum.f128(fp128, fp128) [[READNONE_INTRINSIC]]

__builtin_hypot(f,f);      __builtin_hypotf(f,f);     __builtin_hypotl(f,f); __builtin_hypotf128(f,f);

// NO__ERRNO: declare double @hypot(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare float @hypotf(float noundef, float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @hypotl(x86_fp80 noundef, x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @hypotf128(fp128 noundef, fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @hypot(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @hypotf(float noundef, float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @hypotl(x86_fp80 noundef, x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @hypotf128(fp128 noundef, fp128 noundef) [[NOT_READNONE]]

__builtin_ilogb(f);      __builtin_ilogbf(f);     __builtin_ilogbl(f); __builtin_ilogbf128(f);

// NO__ERRNO: declare i32 @ilogb(double noundef) [[READNONE]]
// NO__ERRNO: declare i32 @ilogbf(float noundef) [[READNONE]]
// NO__ERRNO: declare i32 @ilogbl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare i32 @ilogbf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare i32 @ilogb(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i32 @ilogbf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i32 @ilogbl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i32 @ilogbf128(fp128 noundef) [[NOT_READNONE]]

__builtin_lgamma(f);     __builtin_lgammaf(f);    __builtin_lgammal(f); __builtin_lgammaf128(f);

// NO__ERRNO: declare double @lgamma(double noundef) [[NOT_READNONE]]
// NO__ERRNO: declare float @lgammaf(float noundef) [[NOT_READNONE]]
// NO__ERRNO: declare x86_fp80 @lgammal(x86_fp80 noundef) [[NOT_READNONE]]
// NO__ERRNO: declare fp128 @lgammaf128(fp128 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare double @lgamma(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @lgammaf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @lgammal(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @lgammaf128(fp128 noundef) [[NOT_READNONE]]

__builtin_llrint(f);     __builtin_llrintf(f);    __builtin_llrintl(f); __builtin_llrintf128(f);

// NO__ERRNO: declare i64 @llvm.llrint.i64.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare i64 @llvm.llrint.i64.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare i64 @llvm.llrint.i64.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare i64 @llvm.llrint.i64.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare i64 @llrint(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @llrintf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @llrintl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @llrintf128(fp128 noundef) [[NOT_READNONE]]

__builtin_llround(f);    __builtin_llroundf(f);   __builtin_llroundl(f); __builtin_llroundf128(f);

// NO__ERRNO: declare i64 @llvm.llround.i64.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare i64 @llvm.llround.i64.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare i64 @llvm.llround.i64.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare i64 @llvm.llround.i64.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare i64 @llround(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @llroundf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @llroundl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @llroundf128(fp128 noundef) [[NOT_READNONE]]

__builtin_log(f);        __builtin_logf(f);       __builtin_logl(f); __builtin_logf128(f);

// NO__ERRNO: declare double @llvm.log.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.log.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.log.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.log.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @log(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @logf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @logl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @logf128(fp128 noundef) [[NOT_READNONE]]

__builtin_log10(f);      __builtin_log10f(f);     __builtin_log10l(f); __builtin_log10f128(f);

// NO__ERRNO: declare double @llvm.log10.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.log10.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.log10.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.log10.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @log10(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @log10f(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @log10l(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @log10f128(fp128 noundef) [[NOT_READNONE]]

__builtin_log1p(f);      __builtin_log1pf(f);     __builtin_log1pl(f); __builtin_log1pf128(f);

// NO__ERRNO: declare double @log1p(double noundef) [[READNONE]]
// NO__ERRNO: declare float @log1pf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @log1pl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @log1pf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @log1p(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @log1pf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @log1pl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @log1pf128(fp128 noundef) [[NOT_READNONE]]

__builtin_log2(f);       __builtin_log2f(f);      __builtin_log2l(f); __builtin_log2f128(f);

// NO__ERRNO: declare double @llvm.log2.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.log2.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.log2.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.log2.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @log2(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @log2f(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @log2l(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @log2f128(fp128 noundef) [[NOT_READNONE]]

__builtin_logb(f);       __builtin_logbf(f);      __builtin_logbl(f); __builtin_logbf128(f);

// NO__ERRNO: declare double @logb(double noundef) [[READNONE]]
// NO__ERRNO: declare float @logbf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @logbl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @logbf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @logb(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @logbf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @logbl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @logbf128(fp128 noundef) [[NOT_READNONE]]

__builtin_lrint(f);      __builtin_lrintf(f);     __builtin_lrintl(f); __builtin_lrintf128(f);

// NO__ERRNO: declare i64 @llvm.lrint.i64.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare i64 @llvm.lrint.i64.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare i64 @llvm.lrint.i64.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare i64 @llvm.lrint.i64.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare i64 @lrint(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @lrintf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @lrintl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @lrintf128(fp128 noundef) [[NOT_READNONE]]

__builtin_lround(f);     __builtin_lroundf(f);    __builtin_lroundl(f);  __builtin_lroundf128(f);

// NO__ERRNO: declare i64 @llvm.lround.i64.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare i64 @llvm.lround.i64.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare i64 @llvm.lround.i64.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare i64 @llvm.lround.i64.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare i64 @lround(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @lroundf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @lroundl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare i64 @lroundf128(fp128 noundef) [[NOT_READNONE]]

__builtin_nearbyint(f);  __builtin_nearbyintf(f); __builtin_nearbyintl(f); __builtin_nearbyintf128(f);

// NO__ERRNO: declare double @llvm.nearbyint.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.nearbyint.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.nearbyint.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.nearbyint.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.nearbyint.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.nearbyint.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.nearbyint.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare fp128 @llvm.nearbyint.f128(fp128) [[READNONE_INTRINSIC]]

__builtin_nextafter(f,f);  __builtin_nextafterf(f,f); __builtin_nextafterl(f,f); __builtin_nextafterf128(f,f);

// NO__ERRNO: declare double @nextafter(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare float @nextafterf(float noundef, float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @nextafterl(x86_fp80 noundef, x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @nextafterf128(fp128 noundef, fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @nextafter(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @nextafterf(float noundef, float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @nextafterl(x86_fp80 noundef, x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @nextafterf128(fp128 noundef, fp128 noundef) [[NOT_READNONE]]

__builtin_nexttoward(f,f); __builtin_nexttowardf(f,f);__builtin_nexttowardl(f,f); __builtin_nexttowardf128(f,f);

// NO__ERRNO: declare double @nexttoward(double noundef, x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare float @nexttowardf(float noundef, x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @nexttowardl(x86_fp80 noundef, x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @nexttowardf128(fp128 noundef, fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @nexttoward(double noundef, x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @nexttowardf(float noundef, x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @nexttowardl(x86_fp80 noundef, x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @nexttowardf128(fp128 noundef, fp128 noundef) [[NOT_READNONE]]

__builtin_remainder(f,f);  __builtin_remainderf(f,f); __builtin_remainderl(f,f); __builtin_remainderf128(f,f);

// NO__ERRNO: declare double @remainder(double noundef, double noundef) [[READNONE]]
// NO__ERRNO: declare float @remainderf(float noundef, float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @remainderl(x86_fp80 noundef, x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @remainderf128(fp128 noundef, fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @remainder(double noundef, double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @remainderf(float noundef, float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @remainderl(x86_fp80 noundef, x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @remainderf128(fp128 noundef, fp128 noundef) [[NOT_READNONE]]

__builtin_remquo(f,f,i);  __builtin_remquof(f,f,i); __builtin_remquol(f,f,i); __builtin_remquof128(f,f,i);

// NO__ERRNO: declare double @remquo(double noundef, double noundef, ptr noundef) [[NOT_READNONE]]
// NO__ERRNO: declare float @remquof(float noundef, float noundef, ptr noundef) [[NOT_READNONE]]
// NO__ERRNO: declare x86_fp80 @remquol(x86_fp80 noundef, x86_fp80 noundef, ptr noundef) [[NOT_READNONE]]
// NO__ERRNO: declare fp128 @remquof128(fp128 noundef, fp128 noundef, ptr noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare double @remquo(double noundef, double noundef, ptr noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @remquof(float noundef, float noundef, ptr noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @remquol(x86_fp80 noundef, x86_fp80 noundef, ptr noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @remquof128(fp128 noundef, fp128 noundef, ptr noundef) [[NOT_READNONE]]

__builtin_rint(f);       __builtin_rintf(f);      __builtin_rintl(f); __builtin_rintf128(f);

// NO__ERRNO: declare double @llvm.rint.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.rint.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.rint.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.rint.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.rint.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.rint.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.rint.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare fp128 @llvm.rint.f128(fp128) [[READNONE_INTRINSIC]]

__builtin_round(f);      __builtin_roundf(f);     __builtin_roundl(f); __builtin_roundf128(f);

// NO__ERRNO: declare double @llvm.round.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.round.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.round.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.round.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.round.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.round.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.round.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare fp128 @llvm.round.f128(fp128) [[READNONE_INTRINSIC]]

__builtin_scalbln(f,f);    __builtin_scalblnf(f,f);   __builtin_scalblnl(f,f); __builtin_scalblnf128(f,f);

// NO__ERRNO: declare double @scalbln(double noundef, i64 noundef) [[READNONE]]
// NO__ERRNO: declare float @scalblnf(float noundef, i64 noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @scalblnl(x86_fp80 noundef, i64 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @scalblnf128(fp128 noundef, i64 noundef) [[READNONE]]
// HAS_ERRNO: declare double @scalbln(double noundef, i64 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @scalblnf(float noundef, i64 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @scalblnl(x86_fp80 noundef, i64 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @scalblnf128(fp128 noundef, i64 noundef) [[NOT_READNONE]]

__builtin_scalbn(f,f);     __builtin_scalbnf(f,f);    __builtin_scalbnl(f,f); __builtin_scalbnf128(f,f);

// NO__ERRNO: declare double @scalbn(double noundef, i32 noundef) [[READNONE]]
// NO__ERRNO: declare float @scalbnf(float noundef, i32 noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @scalbnl(x86_fp80 noundef, i32 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @scalbnf128(fp128 noundef, i32 noundef) [[READNONE]]
// HAS_ERRNO: declare double @scalbn(double noundef, i32 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @scalbnf(float noundef, i32 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @scalbnl(x86_fp80 noundef, i32 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @scalbnf128(fp128 noundef, i32 noundef) [[NOT_READNONE]]

__builtin_sin(f);        __builtin_sinf(f);       __builtin_sinl(f); __builtin_sinf128(f);

// NO__ERRNO: declare double @llvm.sin.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.sin.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.sin.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.sin.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @sin(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @sinf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @sinl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @sinf128(fp128 noundef) [[NOT_READNONE]]

__builtin_sinh(f);       __builtin_sinhf(f);      __builtin_sinhl(f); __builtin_sinhf128(f);

// NO__ERRNO: declare double @sinh(double noundef) [[READNONE]]
// NO__ERRNO: declare float @sinhf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @sinhl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @sinhf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @sinh(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @sinhf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @sinhl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @sinhf128(fp128 noundef) [[NOT_READNONE]]

__builtin_sqrt(f);       __builtin_sqrtf(f);      __builtin_sqrtl(f); __builtin_sqrtf128(f);

// NO__ERRNO: declare double @llvm.sqrt.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.sqrt.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.sqrt.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.sqrt.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @sqrt(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @sqrtf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @sqrtl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @sqrtf128(fp128 noundef) [[NOT_READNONE]]

__builtin_tan(f);        __builtin_tanf(f);       __builtin_tanl(f); __builtin_tanf128(f);

// NO__ERRNO: declare double @llvm.tan.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.tan.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.tan.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.tan.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @tan(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @tanf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @tanl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @tanf128(fp128 noundef) [[NOT_READNONE]]

__builtin_tanh(f);       __builtin_tanhf(f);      __builtin_tanhl(f); __builtin_tanhf128(f);

// NO__ERRNO: declare double @tanh(double noundef) [[READNONE]]
// NO__ERRNO: declare float @tanhf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @tanhl(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @tanhf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @tanh(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @tanhf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @tanhl(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @tanhf128(fp128 noundef) [[NOT_READNONE]]

__builtin_tgamma(f);     __builtin_tgammaf(f);    __builtin_tgammal(f); __builtin_tgammaf128(f);

// NO__ERRNO: declare double @tgamma(double noundef) [[READNONE]]
// NO__ERRNO: declare float @tgammaf(float noundef) [[READNONE]]
// NO__ERRNO: declare x86_fp80 @tgammal(x86_fp80 noundef) [[READNONE]]
// NO__ERRNO: declare fp128 @tgammaf128(fp128 noundef) [[READNONE]]
// HAS_ERRNO: declare double @tgamma(double noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare float @tgammaf(float noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare x86_fp80 @tgammal(x86_fp80 noundef) [[NOT_READNONE]]
// HAS_ERRNO: declare fp128 @tgammaf128(fp128 noundef) [[NOT_READNONE]]

__builtin_trunc(f);      __builtin_truncf(f);     __builtin_truncl(f); __builtin_truncf128(f);

// NO__ERRNO: declare double @llvm.trunc.f64(double) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare float @llvm.trunc.f32(float) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare x86_fp80 @llvm.trunc.f80(x86_fp80) [[READNONE_INTRINSIC]]
// NO__ERRNO: declare fp128 @llvm.trunc.f128(fp128) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare double @llvm.trunc.f64(double) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare float @llvm.trunc.f32(float) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare x86_fp80 @llvm.trunc.f80(x86_fp80) [[READNONE_INTRINSIC]]
// HAS_ERRNO: declare fp128 @llvm.trunc.f128(fp128) [[READNONE_INTRINSIC]]
};

// NO__ERRNO: attributes [[READNONE]] = { {{.*}}memory(none){{.*}} }
// NO__ERRNO: attributes [[READNONE_INTRINSIC]] = { {{.*}}memory(none){{.*}} }
// NO__ERRNO: attributes [[NOT_READNONE]] = { nounwind {{.*}} }
// NO__ERRNO: attributes [[PURE]] = { {{.*}}memory(read){{.*}} }

// HAS_ERRNO: attributes [[NOT_READNONE]] = { nounwind {{.*}} }
// HAS_ERRNO: attributes [[READNONE_INTRINSIC]] = { {{.*}}memory(none){{.*}} }
// HAS_ERRNO: attributes [[PURE]] = { {{.*}}memory(read){{.*}} }
// HAS_ERRNO: attributes [[READNONE]] = { {{.*}}memory(none){{.*}} }

// HAS_ERRNO_GNU: attributes [[READNONE_INTRINSIC]] = { {{.*}}memory(none){{.*}} }
// HAS_ERRNO_WIN: attributes [[READNONE_INTRINSIC]] = { {{.*}}memory(none){{.*}} }

