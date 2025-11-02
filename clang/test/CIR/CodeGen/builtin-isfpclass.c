// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int finite(double);

// CHECK: cir.func {{.*}}@test_is_finite
void test_is_finite(__fp16 *H, float F, double D, long double LD) {
    volatile int res;
    res = __builtin_isinf(*H);
    // CHECK: cir.is_fp_class %{{.*}}, 516 : (!cir.f16) -> !cir.bool

    res = __builtin_isinf(F);
    // CHECK: cir.is_fp_class %{{.*}}, 516 : (!cir.float) -> !cir.bool

    res = __builtin_isinf(D);
    // CHECK: cir.is_fp_class %{{.*}}, 516 : (!cir.double) -> !cir.bool

    res = __builtin_isinf(LD);
    // CHECK: cir.is_fp_class %{{.*}}, 516 : (!cir.long_double<!cir.f80>) -> !cir.bool

    res = __builtin_isfinite(*H);
    // CHECK: cir.is_fp_class %{{.*}}, 504 : (!cir.f16) -> !cir.bool
    res = __builtin_isfinite(F);
    // CHECK: cir.is_fp_class %{{.*}}, 504 : (!cir.float) -> !cir.bool
    res = finite(D);
    // CHECK: cir.call @finite(%{{.*}}) nothrow side_effect(const) : (!cir.double) -> !s32i

    res = __builtin_isnormal(*H);
    // CHECK: cir.is_fp_class %{{.*}}, 264 : (!cir.f16) -> !cir.bool
    res = __builtin_isnormal(F);
    // CHECK: cir.is_fp_class %{{.*}}, 264 : (!cir.float) -> !cir.bool

    res = __builtin_issubnormal(F);
    // CHECK: cir.is_fp_class %{{.*}}, 144 : (!cir.float) -> !cir.bool
    res = __builtin_iszero(F);
    // CHECK: cir.is_fp_class %{{.*}}, 96 : (!cir.float) -> !cir.bool
    res = __builtin_issignaling(F);
    // CHECK: cir.is_fp_class %{{.*}}, 1 : (!cir.float) -> !cir.bool
}

_Bool check_isfpclass_finite(float x) {
  return __builtin_isfpclass(x, 504 /*Finite*/);
}

// CHECK: cir.func {{.*}}@check_isfpclass_finite
// CHECK: cir.is_fp_class %{{.*}}, 504 : (!cir.float)

_Bool check_isfpclass_nan_f32(float x) {
  return __builtin_isfpclass(x, 3 /*NaN*/);
}

// CHECK: cir.func {{.*}}@check_isfpclass_nan_f32
// CHECK: cir.is_fp_class %{{.*}}, 3 : (!cir.float)


_Bool check_isfpclass_snan_f64(double x) {
  return __builtin_isfpclass(x, 1 /*SNaN*/);
}

// CHECK: cir.func {{.*}}@check_isfpclass_snan_f64
// CHECK: cir.is_fp_class %{{.*}}, 1 : (!cir.double)


_Bool check_isfpclass_zero_f16(_Float16 x) {
  return __builtin_isfpclass(x, 96 /*Zero*/);
}

// CHECK: cir.func {{.*}}@check_isfpclass_zero_f16
// CHECK: cir.is_fp_class %{{.*}}, 96 : (!cir.f16)

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
