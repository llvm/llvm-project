// RUN: %clang_cc1 -triple x86_64-linux -ffp-exception-behavior=maytrap -w -o - -emit-llvm %s | FileCheck %s

// Test codegen of constrained math builtins.
//
// Test that the constrained intrinsics are picking up the exception
// metadata from the AST instead of the global default from the command line.

#pragma float_control(except, on)

void foo(double *d, float f, float *fp, long double *l, int *i, const char *c, _Float16 h) {
  f = __builtin_fmod(f,f);    f = __builtin_fmodf(f,f);   f =  __builtin_fmodl(f,f); f = __builtin_fmodf128(f,f);

// CHECK: call double @llvm.experimental.constrained.frem.f64(double %{{.*}}, double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.frem.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.frem.f80(x86_fp80 %{{.*}}, x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.frem.f128(fp128 %{{.*}}, fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_pow(f,f);        __builtin_powf(f,f);       __builtin_powl(f,f); __builtin_powf128(f,f);

// CHECK: call double @llvm.experimental.constrained.pow.f64(double %{{.*}}, double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.pow.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.pow.f80(x86_fp80 %{{.*}}, x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.pow.f128(fp128 %{{.*}}, fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_powi(f,f);        __builtin_powif(f,f);       __builtin_powil(f,f);

// CHECK: call double @llvm.experimental.constrained.powi.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.powi.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.powi.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")


  h = __builtin_ldexpf16(h, *i);  *d = __builtin_ldexp(*d, *i);        f = __builtin_ldexpf(f, *i);       __builtin_ldexpl(*l, *i);

// CHECK: call half @llvm.experimental.constrained.ldexp.f16.i32(half %{{.*}}, i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call double @llvm.experimental.constrained.ldexp.f64.i32(double %{{.*}}, i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.ldexp.f32.i32(float %{{.*}}, i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.ldexp.f80.i32(x86_fp80 %{{.*}}, i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_acos(f);        __builtin_acosf(f);       __builtin_acosl(f); __builtin_acosf128(f);

// CHECK: call double @llvm.experimental.constrained.acos.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.acos.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.acos.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.acos.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

__builtin_asin(f);        __builtin_asinf(f);       __builtin_asinl(f); __builtin_asinf128(f);

// CHECK: call double @llvm.experimental.constrained.asin.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.asin.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.asin.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.asin.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

__builtin_atan(f);        __builtin_atanf(f);       __builtin_atanl(f); __builtin_atanf128(f);

// CHECK: call double @llvm.experimental.constrained.atan.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.atan.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.atan.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.atan.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

__builtin_atan2(f,f);        __builtin_atan2f(f,f);       __builtin_atan2l(f,f); __builtin_atan2f128(f,f);

// CHECK: call double @llvm.experimental.constrained.atan2.f64(double %{{.*}}, double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.atan2.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.atan2.f80(x86_fp80 %{{.*}}, x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.atan2.f128(fp128 %{{.*}}, fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_ceil(f);       __builtin_ceilf(f);      __builtin_ceill(f); __builtin_ceilf128(f);

// CHECK: call double @llvm.experimental.constrained.ceil.f64(double %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.ceil.f32(float %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.ceil.f80(x86_fp80 %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.ceil.f128(fp128 %{{.*}}, metadata !"fpexcept.strict")

  __builtin_cos(f);        __builtin_cosf(f);       __builtin_cosl(f); __builtin_cosf128(f);

// CHECK: call double @llvm.experimental.constrained.cos.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.cos.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.cos.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.cos.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_cosh(f);        __builtin_coshf(f);       __builtin_coshl(f); __builtin_coshf128(f);

// CHECK: call double @llvm.experimental.constrained.cosh.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.cosh.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.cosh.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.cosh.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_exp(f);        __builtin_expf(f);       __builtin_expl(f); __builtin_expf128(f);

// CHECK: call double @llvm.experimental.constrained.exp.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.exp.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.exp.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.exp.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_exp2(f);       __builtin_exp2f(f);      __builtin_exp2l(f); __builtin_exp2f128(f);

// CHECK: call double @llvm.experimental.constrained.exp2.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.exp2.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.exp2.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.exp2.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_exp10(f);       __builtin_exp10f(f);      __builtin_exp10l(f); __builtin_exp10f128(f);

// CHECK: call double @exp10(double noundef %{{.*}})
// CHECK: call float @exp10f(float noundef %{{.*}})
// CHECK: call x86_fp80 @exp10l(x86_fp80 noundef %{{.*}})
// CHECK: call fp128 @exp10f128(fp128 noundef %{{.*}})

  __builtin_floor(f);      __builtin_floorf(f);     __builtin_floorl(f); __builtin_floorf128(f);

// CHECK: call double @llvm.experimental.constrained.floor.f64(double %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.floor.f32(float %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.floor.f80(x86_fp80 %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.floor.f128(fp128 %{{.*}}, metadata !"fpexcept.strict")

  __builtin_fma(f,f,f);        __builtin_fmaf(f,f,f);       __builtin_fmal(f,f,f);  __builtin_fmaf128(f,f,f); __builtin_fmaf16(f,f,f);

// CHECK: call double @llvm.experimental.constrained.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.fma.f80(x86_fp80 %{{.*}}, x86_fp80 %{{.*}}, x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.fma.f128(fp128 %{{.*}}, fp128 %{{.*}}, fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call half @llvm.experimental.constrained.fma.f16(half %{{.*}}, half %{{.*}}, half %{{.*}}, metadata !"fpexcept.strict")

  __builtin_fmax(f,f);       __builtin_fmaxf(f,f);      __builtin_fmaxl(f,f); __builtin_fmaxf128(f,f);

// CHECK: call double @llvm.experimental.constrained.maxnum.f64(double %{{.*}}, double %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.maxnum.f32(float %{{.*}}, float %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.maxnum.f80(x86_fp80 %{{.*}}, x86_fp80 %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.maxnum.f128(fp128 %{{.*}}, fp128 %{{.*}}, metadata !"fpexcept.strict")

  __builtin_fmin(f,f);       __builtin_fminf(f,f);      __builtin_fminl(f,f); __builtin_fminf128(f,f);

// CHECK: call double @llvm.experimental.constrained.minnum.f64(double %{{.*}}, double %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.minnum.f32(float %{{.*}}, float %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.minnum.f80(x86_fp80 %{{.*}}, x86_fp80 %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.minnum.f128(fp128 %{{.*}}, fp128 %{{.*}}, metadata !"fpexcept.strict")

  __builtin_llrint(f);     __builtin_llrintf(f);    __builtin_llrintl(f); __builtin_llrintf128(f);

// CHECK: call i64 @llvm.experimental.constrained.llrint.i64.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call i64 @llvm.experimental.constrained.llrint.i64.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call i64 @llvm.experimental.constrained.llrint.i64.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call i64 @llvm.experimental.constrained.llrint.i64.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_llround(f);    __builtin_llroundf(f);   __builtin_llroundl(f); __builtin_llroundf128(f);

// CHECK: call i64 @llvm.experimental.constrained.llround.i64.f64(double %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call i64 @llvm.experimental.constrained.llround.i64.f32(float %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call i64 @llvm.experimental.constrained.llround.i64.f80(x86_fp80 %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call i64 @llvm.experimental.constrained.llround.i64.f128(fp128 %{{.*}}, metadata !"fpexcept.strict")

  __builtin_log(f);        __builtin_logf(f);       __builtin_logl(f); __builtin_logf128(f);

// CHECK: call double @llvm.experimental.constrained.log.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.log.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.log.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.log.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_log10(f);      __builtin_log10f(f);     __builtin_log10l(f); __builtin_log10f128(f);

// CHECK: call double @llvm.experimental.constrained.log10.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.log10.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.log10.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.log10.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_log2(f);       __builtin_log2f(f);      __builtin_log2l(f); __builtin_log2f128(f);

// CHECK: call double @llvm.experimental.constrained.log2.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.log2.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.log2.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.log2.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_lrint(f);      __builtin_lrintf(f);     __builtin_lrintl(f); __builtin_lrintf128(f);

// CHECK: call i64 @llvm.experimental.constrained.lrint.i64.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call i64 @llvm.experimental.constrained.lrint.i64.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call i64 @llvm.experimental.constrained.lrint.i64.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call i64 @llvm.experimental.constrained.lrint.i64.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_lround(f);     __builtin_lroundf(f);    __builtin_lroundl(f); __builtin_lroundf128(f);

// CHECK: call i64 @llvm.experimental.constrained.lround.i64.f64(double %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call i64 @llvm.experimental.constrained.lround.i64.f32(float %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call i64 @llvm.experimental.constrained.lround.i64.f80(x86_fp80 %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call i64 @llvm.experimental.constrained.lround.i64.f128(fp128 %{{.*}}, metadata !"fpexcept.strict")

  __builtin_nearbyint(f);  __builtin_nearbyintf(f); __builtin_nearbyintl(f); __builtin_nearbyintf128(f);

// CHECK: call double @llvm.experimental.constrained.nearbyint.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.nearbyint.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.nearbyint.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.nearbyint.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_rint(f);       __builtin_rintf(f);      __builtin_rintl(f); __builtin_rintf128(f);

// CHECK: call double @llvm.experimental.constrained.rint.f64(double %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.rint.f32(float %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.rint.f80(x86_fp80 %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.rint.f128(fp128 %{{.*}}, metadata !"fpexcept.strict")

  __builtin_round(f);      __builtin_roundf(f);     __builtin_roundl(f); __builtin_roundf128(f);

// CHECK: call double @llvm.experimental.constrained.round.f64(double %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.round.f32(float %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.round.f80(x86_fp80 %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.round.f128(fp128 %{{.*}}, metadata !"fpexcept.strict")

  __builtin_sin(f);        __builtin_sinf(f);       __builtin_sinl(f); __builtin_sinf128(f);

// CHECK: call double @llvm.experimental.constrained.sin.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.sin.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.sin.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.sin.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_sinh(f);        __builtin_sinhf(f);       __builtin_sinhl(f); __builtin_sinhf128(f);

// CHECK: call double @llvm.experimental.constrained.sinh.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.sinh.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.sinh.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.sinh.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_sqrt(f);       __builtin_sqrtf(f);      __builtin_sqrtl(f); __builtin_sqrtf128(f);

// CHECK: call double @llvm.experimental.constrained.sqrt.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.sqrt.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.sqrt.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.sqrt.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_tan(f);        __builtin_tanf(f);       __builtin_tanl(f); __builtin_tanf128(f);

// CHECK: call double @llvm.experimental.constrained.tan.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.tan.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.tan.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.tan.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_tanh(f);        __builtin_tanhf(f);       __builtin_tanhl(f); __builtin_tanhf128(f);

// CHECK: call double @llvm.experimental.constrained.tanh.f64(double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.tanh.f32(float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.tanh.f80(x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.tanh.f128(fp128 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

  __builtin_trunc(f);      __builtin_truncf(f);     __builtin_truncl(f); __builtin_truncf128(f);

// CHECK: call double @llvm.experimental.constrained.trunc.f64(double %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.trunc.f32(float %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call x86_fp80 @llvm.experimental.constrained.trunc.f80(x86_fp80 %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call fp128 @llvm.experimental.constrained.trunc.f128(fp128 %{{.*}}, metadata !"fpexcept.strict")
};

// CHECK: declare double @llvm.experimental.constrained.frem.f64(double, double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.frem.f32(float, float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.frem.f80(x86_fp80, x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.frem.f128(fp128, fp128, metadata, metadata)

// CHECK: declare double @llvm.experimental.constrained.pow.f64(double, double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.pow.f32(float, float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.pow.f80(x86_fp80, x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.pow.f128(fp128, fp128, metadata, metadata)

// CHECK: declare double @llvm.experimental.constrained.powi.f64(double, i32, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.powi.f32(float, i32, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.powi.f80(x86_fp80, i32, metadata, metadata)

// CHECK: declare half @llvm.experimental.constrained.ldexp.f16.i32(half, i32, metadata, metadata)
// CHECK: declare double @llvm.experimental.constrained.ldexp.f64.i32(double, i32, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.ldexp.f32.i32(float, i32, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.ldexp.f80.i32(x86_fp80, i32, metadata, metadata)

// CHECK: declare double @llvm.experimental.constrained.ceil.f64(double, metadata)
// CHECK: declare float @llvm.experimental.constrained.ceil.f32(float, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.ceil.f80(x86_fp80, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.ceil.f128(fp128, metadata)

// CHECK: declare double @llvm.experimental.constrained.cos.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.cos.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.cos.f80(x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.cos.f128(fp128, metadata, metadata)

// CHECK: declare double @llvm.experimental.constrained.exp.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.exp.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.exp.f80(x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.exp.f128(fp128, metadata, metadata)

// CHECK: declare double @llvm.experimental.constrained.exp2.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.exp2.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.exp2.f80(x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.exp2.f128(fp128, metadata, metadata)

// CHECK: declare double @exp10(double noundef)
// CHECK: declare float @exp10f(float noundef)
// CHECK: declare x86_fp80 @exp10l(x86_fp80 noundef)
// CHECK: declare fp128 @exp10f128(fp128 noundef)

// CHECK: declare double @llvm.experimental.constrained.floor.f64(double, metadata)
// CHECK: declare float @llvm.experimental.constrained.floor.f32(float, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.floor.f80(x86_fp80, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.floor.f128(fp128, metadata)

// CHECK: declare double @llvm.experimental.constrained.fma.f64(double, double, double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.fma.f80(x86_fp80, x86_fp80, x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.fma.f128(fp128, fp128, fp128, metadata, metadata)

// CHECK: declare double @llvm.experimental.constrained.maxnum.f64(double, double, metadata)
// CHECK: declare float @llvm.experimental.constrained.maxnum.f32(float, float, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.maxnum.f80(x86_fp80, x86_fp80, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.maxnum.f128(fp128, fp128, metadata)

// CHECK: declare double @llvm.experimental.constrained.minnum.f64(double, double, metadata)
// CHECK: declare float @llvm.experimental.constrained.minnum.f32(float, float, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.minnum.f80(x86_fp80, x86_fp80, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.minnum.f128(fp128, fp128, metadata)

// CHECK: declare i64 @llvm.experimental.constrained.llrint.i64.f64(double, metadata, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.llrint.i64.f32(float, metadata, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.llrint.i64.f80(x86_fp80, metadata, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.llrint.i64.f128(fp128, metadata, metadata)

// CHECK: declare i64 @llvm.experimental.constrained.llround.i64.f64(double, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.llround.i64.f32(float, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.llround.i64.f80(x86_fp80, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.llround.i64.f128(fp128, metadata)

// CHECK: declare double @llvm.experimental.constrained.log.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.log.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.log.f80(x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.log.f128(fp128, metadata, metadata)

// CHECK: declare double @llvm.experimental.constrained.log10.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.log10.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.log10.f80(x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.log10.f128(fp128, metadata, metadata)

// CHECK: declare double @llvm.experimental.constrained.log2.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.log2.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.log2.f80(x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.log2.f128(fp128, metadata, metadata)

// CHECK: declare i64 @llvm.experimental.constrained.lrint.i64.f64(double, metadata, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.lrint.i64.f32(float, metadata, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.lrint.i64.f80(x86_fp80, metadata, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.lrint.i64.f128(fp128, metadata, metadata)

// CHECK: declare i64 @llvm.experimental.constrained.lround.i64.f64(double, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.lround.i64.f32(float, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.lround.i64.f80(x86_fp80, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.lround.i64.f128(fp128, metadata)

// CHECK: declare double @llvm.experimental.constrained.nearbyint.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.nearbyint.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.nearbyint.f80(x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.nearbyint.f128(fp128, metadata, metadata)

// CHECK: declare double @llvm.experimental.constrained.rint.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.rint.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.rint.f80(x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.rint.f128(fp128, metadata, metadata)

// CHECK: declare double @llvm.experimental.constrained.round.f64(double, metadata)
// CHECK: declare float @llvm.experimental.constrained.round.f32(float, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.round.f80(x86_fp80, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.round.f128(fp128, metadata)

// CHECK: declare double @llvm.experimental.constrained.sin.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.sin.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.sin.f80(x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.sin.f128(fp128, metadata, metadata)

// CHECK: declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.sqrt.f80(x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.sqrt.f128(fp128, metadata, metadata)

// CHECK: declare double @llvm.experimental.constrained.tan.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.tan.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.tan.f80(x86_fp80, metadata, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.tan.f128(fp128, metadata, metadata)

// CHECK: declare double @llvm.experimental.constrained.trunc.f64(double, metadata)
// CHECK: declare float @llvm.experimental.constrained.trunc.f32(float, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.trunc.f80(x86_fp80, metadata)
// CHECK: declare fp128 @llvm.experimental.constrained.trunc.f128(fp128, metadata)

#pragma STDC FP_CONTRACT ON
void bar(float f) {
  f * f + f;
  (double)f * f - f;
  (long double)-f * f + f;
  -(f * f) - f;
  f + -(f * f);

  // CHECK: call float @llvm.experimental.constrained.fmuladd.f32(float %{{.*}}, float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: fneg
  // CHECK: call double @llvm.experimental.constrained.fmuladd.f64(double %{{.*}}, double %{{.*}}, double %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: fneg
  // CHECK: call x86_fp80 @llvm.experimental.constrained.fmuladd.f80(x86_fp80 %{{.*}}, x86_fp80 %{{.*}}, x86_fp80 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call float @llvm.experimental.constrained.fmuladd.f32(float %{{.*}}, float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: fneg
  // CHECK: call float @llvm.experimental.constrained.fmuladd.f32(float %{{.*}}, float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
};
