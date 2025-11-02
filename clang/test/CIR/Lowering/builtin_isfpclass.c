// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

int finite(double);

// CHECK: define {{.*}}@test_is_finite
void test_is_finite(__fp16 *H, float F, double D, long double LD) {
    volatile int res;
    res = __builtin_isinf(*H);
    // CHECK: call i1 @llvm.is.fpclass.f16(half %{{.*}}, i32 516)
    res = __builtin_isinf(F);
    // CHECK: call i1 @llvm.is.fpclass.f32(float %{{.*}}, i32 516)
    res = __builtin_isinf(D);
    // CHECK: call i1 @llvm.is.fpclass.f64(double %{{.*}}, i32 516)
    res = __builtin_isinf(LD);
    // CHECK: call i1 @llvm.is.fpclass.f80(x86_fp80 %{{.*}}, i32 516)

    res = __builtin_isfinite(*H);
    // CHECK: call i1 @llvm.is.fpclass.f16(half %{{.*}}, i32 504)
    res = __builtin_isfinite(F);
    // CHECK: call i1 @llvm.is.fpclass.f32(float %{{.*}}, i32 504)
    res = finite(D);
    // CHECK: call i32 @finite(double %{{.*}})

    res = __builtin_isnormal(*H);
    // CHECK: call i1 @llvm.is.fpclass.f16(half %{{.*}}, i32 264)
    res = __builtin_isnormal(F);
    // CHECK: call i1 @llvm.is.fpclass.f32(float %{{.*}}, i32 264)

    res = __builtin_issubnormal(F);
    // CHECK: call i1 @llvm.is.fpclass.f32(float %{{.*}}, i32 144)
    res = __builtin_iszero(F);
    // CHECK: call i1 @llvm.is.fpclass.f32(float %{{.*}}, i32 96)
    res = __builtin_issignaling(F);
    // CHECK: call i1 @llvm.is.fpclass.f32(float %{{.*}}, i32 1)
}

_Bool check_isfpclass_finite(float x) {
  return __builtin_isfpclass(x, 504 /*Finite*/);
}

// CHECK: define {{.*}}@check_isfpclass_finite
// CHECK: call i1 @llvm.is.fpclass.f32(float %{{.*}}, i32 504)

_Bool check_isfpclass_nan_f32(float x) {
  return __builtin_isfpclass(x, 3 /*NaN*/);
}

// CHECK: define {{.*}}@check_isfpclass_nan_f32
// CHECK: call i1 @llvm.is.fpclass.f32(float %{{.*}}, i32 3)

_Bool check_isfpclass_snan_f64(double x) {
  return __builtin_isfpclass(x, 1 /*SNaN*/);
}

// CHECK: define {{.*}}@check_isfpclass_snan_f64
// CHECK: call i1 @llvm.is.fpclass.f64(double %{{.*}}, i32 1)


_Bool check_isfpclass_zero_f16(_Float16 x) {
  return __builtin_isfpclass(x, 96 /*Zero*/);
}

// CHECK: define {{.*}}@check_isfpclass_zero_f16
// CHECK: call i1 @llvm.is.fpclass.f16(half %{{.*}}, i32 96)

// Update when we support FP pragma in functions.

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
