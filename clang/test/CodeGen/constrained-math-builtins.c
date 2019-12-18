// RUN: %clang_cc1 -triple x86_64-linux -ffp-exception-behavior=strict -w -S -o - -emit-llvm %s | FileCheck %s

// Test codegen of constrained math builtins.

void foo(double *d, float f, float *fp, long double *l, int *i, const char *c) {
  f = __builtin_fmod(f,f);    f = __builtin_fmodf(f,f);   f =  __builtin_fmodl(f,f);

// CHECK: declare double @llvm.experimental.constrained.frem.f64(double, double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.frem.f32(float, float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.frem.f80(x86_fp80, x86_fp80, metadata, metadata)

  __builtin_pow(f,f);        __builtin_powf(f,f);       __builtin_powl(f,f);

// CHECK: declare double @llvm.experimental.constrained.pow.f64(double, double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.pow.f32(float, float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.pow.f80(x86_fp80, x86_fp80, metadata, metadata)

  __builtin_powi(f,f);        __builtin_powif(f,f);       __builtin_powil(f,f);

// CHECK: declare double @llvm.experimental.constrained.powi.f64(double, i32, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.powi.f32(float, i32, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.powi.f80(x86_fp80, i32, metadata, metadata)

  __builtin_ceil(f);       __builtin_ceilf(f);      __builtin_ceill(f);

// CHECK: declare double @llvm.experimental.constrained.ceil.f64(double, metadata)
// CHECK: declare float @llvm.experimental.constrained.ceil.f32(float, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.ceil.f80(x86_fp80, metadata)

  __builtin_cos(f);        __builtin_cosf(f);       __builtin_cosl(f); 

// CHECK: declare double @llvm.experimental.constrained.cos.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.cos.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.cos.f80(x86_fp80, metadata, metadata)

  __builtin_exp(f);        __builtin_expf(f);       __builtin_expl(f);

// CHECK: declare double @llvm.experimental.constrained.exp.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.exp.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.exp.f80(x86_fp80, metadata, metadata)

  __builtin_exp2(f);       __builtin_exp2f(f);      __builtin_exp2l(f); 

// CHECK: declare double @llvm.experimental.constrained.exp2.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.exp2.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.exp2.f80(x86_fp80, metadata, metadata)

  __builtin_floor(f);      __builtin_floorf(f);     __builtin_floorl(f);

// CHECK: declare double @llvm.experimental.constrained.floor.f64(double, metadata)
// CHECK: declare float @llvm.experimental.constrained.floor.f32(float, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.floor.f80(x86_fp80, metadata)

  __builtin_fma(f,f,f);        __builtin_fmaf(f,f,f);       __builtin_fmal(f,f,f);

// CHECK: declare double @llvm.experimental.constrained.fma.f64(double, double, double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.fma.f80(x86_fp80, x86_fp80, x86_fp80, metadata, metadata)

  __builtin_fmax(f,f);       __builtin_fmaxf(f,f);      __builtin_fmaxl(f,f);

// CHECK: declare double @llvm.experimental.constrained.maxnum.f64(double, double, metadata)
// CHECK: declare float @llvm.experimental.constrained.maxnum.f32(float, float, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.maxnum.f80(x86_fp80, x86_fp80, metadata)

  __builtin_fmin(f,f);       __builtin_fminf(f,f);      __builtin_fminl(f,f);

// CHECK: declare double @llvm.experimental.constrained.minnum.f64(double, double, metadata)
// CHECK: declare float @llvm.experimental.constrained.minnum.f32(float, float, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.minnum.f80(x86_fp80, x86_fp80, metadata)

  __builtin_llrint(f);     __builtin_llrintf(f);    __builtin_llrintl(f);

// CHECK: declare i64 @llvm.experimental.constrained.llrint.i64.f64(double, metadata, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.llrint.i64.f32(float, metadata, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.llrint.i64.f80(x86_fp80, metadata, metadata)

  __builtin_llround(f);    __builtin_llroundf(f);   __builtin_llroundl(f);

// CHECK: declare i64 @llvm.experimental.constrained.llround.i64.f64(double, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.llround.i64.f32(float, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.llround.i64.f80(x86_fp80, metadata)

  __builtin_log(f);        __builtin_logf(f);       __builtin_logl(f);

// CHECK: declare double @llvm.experimental.constrained.log.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.log.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.log.f80(x86_fp80, metadata, metadata)

  __builtin_log10(f);      __builtin_log10f(f);     __builtin_log10l(f);

// CHECK: declare double @llvm.experimental.constrained.log10.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.log10.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.log10.f80(x86_fp80, metadata, metadata)

  __builtin_log2(f);       __builtin_log2f(f);      __builtin_log2l(f);

// CHECK: declare double @llvm.experimental.constrained.log2.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.log2.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.log2.f80(x86_fp80, metadata, metadata)

  __builtin_lrint(f);      __builtin_lrintf(f);     __builtin_lrintl(f);

// CHECK: declare i64 @llvm.experimental.constrained.lrint.i64.f64(double, metadata, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.lrint.i64.f32(float, metadata, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.lrint.i64.f80(x86_fp80, metadata, metadata)

  __builtin_lround(f);     __builtin_lroundf(f);    __builtin_lroundl(f);

// CHECK: declare i64 @llvm.experimental.constrained.lround.i64.f64(double, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.lround.i64.f32(float, metadata)
// CHECK: declare i64 @llvm.experimental.constrained.lround.i64.f80(x86_fp80, metadata)

  __builtin_nearbyint(f);  __builtin_nearbyintf(f); __builtin_nearbyintl(f);

// CHECK: declare double @llvm.experimental.constrained.nearbyint.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.nearbyint.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.nearbyint.f80(x86_fp80, metadata, metadata)

  __builtin_rint(f);       __builtin_rintf(f);      __builtin_rintl(f);

// CHECK: declare double @llvm.experimental.constrained.rint.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.rint.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.rint.f80(x86_fp80, metadata, metadata)

  __builtin_round(f);      __builtin_roundf(f);     __builtin_roundl(f);

// CHECK: declare double @llvm.experimental.constrained.round.f64(double, metadata)
// CHECK: declare float @llvm.experimental.constrained.round.f32(float, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.round.f80(x86_fp80, metadata)

  __builtin_sin(f);        __builtin_sinf(f);       __builtin_sinl(f);

// CHECK: declare double @llvm.experimental.constrained.sin.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.sin.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.sin.f80(x86_fp80, metadata, metadata)

  __builtin_sqrt(f);       __builtin_sqrtf(f);      __builtin_sqrtl(f); 

// CHECK: declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)
// CHECK: declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.sqrt.f80(x86_fp80, metadata, metadata)

  __builtin_trunc(f);      __builtin_truncf(f);     __builtin_truncl(f);

// CHECK: declare double @llvm.experimental.constrained.trunc.f64(double, metadata)
// CHECK: declare float @llvm.experimental.constrained.trunc.f32(float, metadata)
// CHECK: declare x86_fp80 @llvm.experimental.constrained.trunc.f80(x86_fp80, metadata)
};

