// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=OGCG
int finite(double);

// CHECK: cir.func {{.*}}@test_is_finite
void test_is_finite(__fp16 *H, float F, double D, long double LD) {
    volatile int res;
    res = __builtin_isinf(*H);
    // CIR: cir.is_fp_class %{{.*}}, "fcInf" : (!cir.f16) -> !cir.bool
    // LLVM: call i1 @llvm.is.fpclass.f16(half {{.*}}, /* (inf) */ i32 516)
    // OGCG: call i1 @llvm.is.fpclass.f16(half {{.*}}, /* (inf) */ i32 516)

    res = __builtin_isinf(F);
    // CIR: cir.is_fp_class %{{.*}}, "fcInf" : (!cir.float) -> !cir.bool
    // LLVM: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (inf) */ i32 516)
    // OGCG: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (inf) */ i32 516)

    res = __builtin_isinf(D);
    // CIR: cir.is_fp_class %{{.*}}, "fcInf" : (!cir.double) -> !cir.bool
    // LLVM: call i1 @llvm.is.fpclass.f64(double {{.*}}, /* (inf) */ i32 516)
    // OGCG: call i1 @llvm.is.fpclass.f64(double {{.*}}, /* (inf) */ i32 516)

    res = __builtin_isinf(LD);
    // CIR: cir.is_fp_class %{{.*}}, "fcInf" : (!cir.long_double<!cir.f80>) -> !cir.bool
    // LLVM: call i1 @llvm.is.fpclass.f80(x86_fp80 {{.*}}, /* (inf) */ i32 516)
    // OGCG: call i1 @llvm.is.fpclass.f80(x86_fp80 {{.*}}, /* (inf) */ i32 516)

    res = __builtin_isfinite(*H);
    // CIR: cir.is_fp_class %{{.*}}, "fcFinite" : (!cir.f16) -> !cir.bool
    // LLVM: call i1 @llvm.is.fpclass.f16(half {{.*}}, /* (zero sub norm) */ i32 504)
    // OGCG: call i1 @llvm.is.fpclass.f16(half {{.*}}, /* (zero sub norm) */ i32 504)

    res = __builtin_isfinite(F);
    // CIR: cir.is_fp_class %{{.*}}, "fcFinite" : (!cir.float) -> !cir.bool
    // LLVM: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (zero sub norm) */ i32 504)
    // OGCG: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (zero sub norm) */ i32 504)

    res = finite(D);
    // CIR: cir.is_fp_class %{{.*}}, "fcFinite" : (!cir.double) -> !cir.bool
    // LLVM: call i1 @llvm.is.fpclass.f64(double {{.*}}, /* (zero sub norm) */ i32 504)
    // OGCG: call i1 @llvm.is.fpclass.f64(double %20, /* (zero sub norm) */ i32 504)
    res = __builtin_isnormal(*H);
    // CIR: cir.is_fp_class %{{.*}}, "fcNormal" : (!cir.f16) -> !cir.bool
    // LLVM: call i1 @llvm.is.fpclass.f16(half {{.*}}, /* (norm) */ i32 264)
    // OGCG: call i1 @llvm.is.fpclass.f16(half {{.*}}, /* (norm) */ i32 264)

    res = __builtin_isnormal(F);
    // CIR: cir.is_fp_class %{{.*}}, "fcNormal" : (!cir.float) -> !cir.bool
    // LLVM: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (norm) */ i32 264)
    // OGCG: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (norm) */ i32 264)

    res = __builtin_issubnormal(F);
    // CIR: cir.is_fp_class %{{.*}}, "fcSubnormal" : (!cir.float) -> !cir.bool
    // LLVM: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (sub) */ i32 144)
    // OGCG: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (sub) */ i32 144)
    res = __builtin_iszero(F);
    // CIR: cir.is_fp_class %{{.*}}, "fcZero" : (!cir.float) -> !cir.bool
    // LLVM: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (zero) */ i32 96)
    // OGCG: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (zero) */ i32 96)
    res = __builtin_issignaling(F);
    // CIR: cir.is_fp_class %{{.*}}, fcSNan : (!cir.float) -> !cir.bool
    // LLVM: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (snan) */ i32 1)
    // OGCG: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (snan) */ i32 1)
}

_Bool check_isfpclass_finite(float x) {
  return __builtin_isfpclass(x, 504 /*Finite*/);
}

// CIR: cir.func {{.*}}@check_isfpclass_finite
// CIR: cir.is_fp_class %{{.*}}, "fcFinite" : (!cir.float)
// LLVM: @check_isfpclass_finite
// LLVM: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (zero sub norm) */ i32 504)
// OGCG: @check_isfpclass_finite
// OGCG: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (zero sub norm) */ i32 504)

_Bool check_isfpclass_nan_f32(float x) {
  return __builtin_isfpclass(x, 3 /*NaN*/);
}

// CIR: cir.func {{.*}}@check_isfpclass_nan_f32
// CIR: cir.is_fp_class %{{.*}}, "fcNan" : (!cir.float)
// LLVM: @check_isfpclass_nan_f32
// LLVM: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (nan) */ i32 3)
// OGCG: @check_isfpclass_nan_f32
// OGCG: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (nan) */ i32 3)


_Bool check_isfpclass_snan_f64(double x) {
  return __builtin_isfpclass(x, 1 /*SNaN*/);
}

// CIR: cir.func {{.*}}@check_isfpclass_snan_f64
// CIR: cir.is_fp_class %{{.*}}, fcSNan : (!cir.double)
// LLVM: @check_isfpclass_snan_f64
// LLVM: call i1 @llvm.is.fpclass.f64(double {{.*}}, /* (snan) */ i32 1)
// OGCG: @check_isfpclass_snan_f64
// OGCG: call i1 @llvm.is.fpclass.f64(double {{.*}}, /* (snan) */ i32 1)


_Bool check_isfpclass_zero_f16(_Float16 x) {
  return __builtin_isfpclass(x, 96 /*Zero*/);
}

// CIR: cir.func {{.*}}@check_isfpclass_zero_f16
// CIR: cir.is_fp_class %{{.*}}, "fcZero" : (!cir.f16)
// LLVM: @check_isfpclass_zero_f16
// LLVM: call i1 @llvm.is.fpclass.f16(half {{.*}}, /* (zero) */ i32 96)
// OGCG: @check_isfpclass_zero_f16
// OGCG: call i1 @llvm.is.fpclass.f16(half {{.*}}, /* (zero) */ i32 96)

_Bool check_isfpclass_snan_neginf(double x) {
  return __builtin_isfpclass(x, 5 /*fcSNan|fcNegInf*/);
}

// CIR: cir.func {{.*}}@check_isfpclass_snan_neginf
// CIR: cir.is_fp_class %{{.*}}, "fcSNan|fcNegInf" : (!cir.double)
// LLVM: @check_isfpclass_snan_neginf
// LLVM: call i1 @llvm.is.fpclass.f64(double {{.*}}, /* (snan ninf) */ i32 5)
// OGCG: @check_isfpclass_snan_neginf
// OGCG: call i1 @llvm.is.fpclass.f64(double {{.*}}, /* (snan ninf) */ i32 5)

_Bool check_isfpclass_multi(float x) {
  return __builtin_isfpclass(x, 1022 /*all but fcSNan (fcQNan still set)*/);
}

// CIR: cir.func {{.*}}@check_isfpclass_multi
// CIR: cir.is_fp_class %{{.*}}, "fcNegative|fcPositive|fcQNan" : (!cir.float)
// LLVM: @check_isfpclass_multi
// LLVM: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (qnan inf zero sub norm) */ i32 1022)
// OGCG: @check_isfpclass_multi
// OGCG: call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (qnan inf zero sub norm) */ i32 1022)

// Update when we support FP pragma in functions and can convert BoolType in prvalue to i1.

// _Bool check_isfpclass_finite_strict(float x) {
// #pragma STDC FENV_ACCESS ON
//   return __builtin_isfpclass(x, 504 /*Finite*/);
// }
// 
// _Bool check_isfpclass_nan_f32_strict(float x) {
// #pragma STDC FENV_ACCESS ON
//   return __builtin_isfpclass(x, 3 /*NaN*/);
// }
// 
// _Bool check_isfpclass_snan_f64_strict(double x) {
// #pragma STDC FENV_ACCESS ON
//   return __builtin_isfpclass(x, 1 /*NaN*/);
// }
// 
// _Bool check_isfpclass_zero_f16_strict(_Float16 x) {
// #pragma STDC FENV_ACCESS ON
//   return __builtin_isfpclass(x, 96 /*Zero*/);
// }
// 
// _Bool check_isnan(float x) {
// #pragma STDC FENV_ACCESS ON
//   return __builtin_isnan(x);
// }
// 
// _Bool check_isinf(float x) {
// #pragma STDC FENV_ACCESS ON
//   return __builtin_isinf(x);
// }
// 
// _Bool check_isfinite(float x) {
// #pragma STDC FENV_ACCESS ON
//   return __builtin_isfinite(x);
// }
// 
// _Bool check_isnormal(float x) {
// #pragma STDC FENV_ACCESS ON
//   return __builtin_isnormal(x);
// }
// 
// typedef float __attribute__((ext_vector_type(4))) float4;
// typedef double __attribute__((ext_vector_type(4))) double4;
// typedef int __attribute__((ext_vector_type(4))) int4;
// typedef long __attribute__((ext_vector_type(4))) long4;
// 
// int4 check_isfpclass_nan_v4f32(float4 x) {
//   return __builtin_isfpclass(x, 3 /*NaN*/);
// }
// 
// int4 check_isfpclass_nan_strict_v4f32(float4 x) {
// #pragma STDC FENV_ACCESS ON
//   return __builtin_isfpclass(x, 3 /*NaN*/);
// }
// 
// long4 check_isfpclass_nan_v4f64(double4 x) {
//   return __builtin_isfpclass(x, 3 /*NaN*/);
// }
