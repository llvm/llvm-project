// RUN: %clang_cc1 -triple=aarch64-gnu-linux -emit-llvm -O1 %s -o - | FileCheck --check-prefix=NO-MATH-ERRNO %s
// RUN: %clang_cc1 -triple=aarch64-gnu-linux -emit-llvm -fmath-errno %s -o - | FileCheck --check-prefix=MATH-ERRNO %s

// NO-MATH-ERRNO-LABEL: @sincos_f32
//      NO-MATH-ERRNO:    [[SINCOS:%.*]] = tail call { float, float } @llvm.sincos.f32(float {{.*}})
// NO-MATH-ERRNO-NEXT:    [[SIN:%.*]] = extractvalue { float, float } [[SINCOS]], 0
// NO-MATH-ERRNO-NEXT:    [[COS:%.*]] = extractvalue { float, float } [[SINCOS]], 1
// NO-MATH-ERRNO-NEXT:    store float [[SIN]], ptr {{.*}}, align 4, !alias.scope [[SINCOS_ALIAS_SCOPE:![0-9]+]]
// NO-MATH-ERRNO-NEXT:    store float [[COS]], ptr {{.*}}, align 4, !noalias [[SINCOS_ALIAS_SCOPE]]
//
// MATH-ERRNO-LABEL: @sincos_f32
//      MATH-ERRNO:    call void @sincosf(
//
void sincos_f32(float x, float* fp0, float* fp1) {
  __builtin_sincosf(x, fp0, fp1);
}

// NO-MATH-ERRNO-LABEL: @sincos_f64
//      NO-MATH-ERRNO:    [[SINCOS:%.*]] = tail call { double, double } @llvm.sincos.f64(double {{.*}})
// NO-MATH-ERRNO-NEXT:    [[SIN:%.*]] = extractvalue { double, double } [[SINCOS]], 0
// NO-MATH-ERRNO-NEXT:    [[COS:%.*]] = extractvalue { double, double } [[SINCOS]], 1
// NO-MATH-ERRNO-NEXT:    store double [[SIN]], ptr {{.*}}, align 8, !alias.scope [[SINCOS_ALIAS_SCOPE:![0-9]+]]
// NO-MATH-ERRNO-NEXT:    store double [[COS]], ptr {{.*}}, align 8, !noalias [[SINCOS_ALIAS_SCOPE]]
//
// MATH-ERRNO-LABEL: @sincos_f64
//      MATH-ERRNO:    call void @sincos(
//
void sincos_f64(double x, double* dp0, double* dp1) {
  __builtin_sincos(x, dp0, dp1);
}

// NO-MATH-ERRNO-LABEL: @sincos_f128
//      NO-MATH-ERRNO:    [[SINCOS:%.*]] = tail call { fp128, fp128 } @llvm.sincos.f128(fp128 {{.*}})
// NO-MATH-ERRNO-NEXT:    [[SIN:%.*]] = extractvalue { fp128, fp128 } [[SINCOS]], 0
// NO-MATH-ERRNO-NEXT:    [[COS:%.*]] = extractvalue { fp128, fp128 } [[SINCOS]], 1
// NO-MATH-ERRNO-NEXT:    store fp128 [[SIN]], ptr {{.*}}, align 16, !alias.scope [[SINCOS_ALIAS_SCOPE:![0-9]+]]
// NO-MATH-ERRNO-NEXT:    store fp128 [[COS]], ptr {{.*}}, align 16, !noalias [[SINCOS_ALIAS_SCOPE]]
//
// MATH-ERRNO-LABEL: @sincos_f128
//      MATH-ERRNO:    call void @sincosl(
//
void sincos_f128(long double x, long double* ldp0, long double* ldp1) {
  __builtin_sincosl(x, ldp0, ldp1);
}
