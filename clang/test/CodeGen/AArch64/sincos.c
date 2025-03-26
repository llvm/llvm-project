// RUN: %clang_cc1 -triple=aarch64-gnu-linux -emit-llvm -O1 %s -o - | FileCheck --check-prefix=NO-MATH-ERRNO %s
// RUN: %clang_cc1 -triple=aarch64-gnu-linux -emit-llvm -fmath-errno %s -o - | FileCheck --check-prefix=MATH-ERRNO %s
// RUN: %clang_cc1 -triple=aarch64-gnu-linux -emit-llvm -ffp-exception-behavior=strict %s -o - | FileCheck --check-prefix=STRICT-FP %s

void sincos(double, double*, double*);
void sincosf(float, float*, float*);
void sincosl(long double, long double*, long double*);

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
// STRICT-FP-LABEL: @sincos_f32
//      STRICT-FP:    call void @sincosf(
//
void sincos_f32(float x, float* fp0, float* fp1) {
  sincosf(x, fp0, fp1);
}

// NO-MATH-ERRNO-LABEL: @sincos_builtin_f32
//      NO-MATH-ERRNO:    [[SINCOS:%.*]] = tail call { float, float } @llvm.sincos.f32(float {{.*}})
// NO-MATH-ERRNO-NEXT:    [[SIN:%.*]] = extractvalue { float, float } [[SINCOS]], 0
// NO-MATH-ERRNO-NEXT:    [[COS:%.*]] = extractvalue { float, float } [[SINCOS]], 1
// NO-MATH-ERRNO-NEXT:    store float [[SIN]], ptr {{.*}}, align 4, !alias.scope [[SINCOS_ALIAS_SCOPE:![0-9]+]]
// NO-MATH-ERRNO-NEXT:    store float [[COS]], ptr {{.*}}, align 4, !noalias [[SINCOS_ALIAS_SCOPE]]
//
// MATH-ERRNO-LABEL: @sincos_builtin_f32
//      MATH-ERRNO:    call void @sincosf(
//
void sincos_builtin_f32(float x, float* fp0, float* fp1) {
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
// STRICT-FP-LABEL: @sincos_f64
//      STRICT-FP:    call void @sincos(
//
void sincos_f64(double x, double* dp0, double* dp1) {
  sincos(x, dp0, dp1);
}

// NO-MATH-ERRNO-LABEL: @sincos_builtin_f64
//      NO-MATH-ERRNO:    [[SINCOS:%.*]] = tail call { double, double } @llvm.sincos.f64(double {{.*}})
// NO-MATH-ERRNO-NEXT:    [[SIN:%.*]] = extractvalue { double, double } [[SINCOS]], 0
// NO-MATH-ERRNO-NEXT:    [[COS:%.*]] = extractvalue { double, double } [[SINCOS]], 1
// NO-MATH-ERRNO-NEXT:    store double [[SIN]], ptr {{.*}}, align 8, !alias.scope [[SINCOS_ALIAS_SCOPE:![0-9]+]]
// NO-MATH-ERRNO-NEXT:    store double [[COS]], ptr {{.*}}, align 8, !noalias [[SINCOS_ALIAS_SCOPE]]
//
// MATH-ERRNO-LABEL: @sincos_builtin_f64
//      MATH-ERRNO:    call void @sincos(
//
void sincos_builtin_f64(double x, double* dp0, double* dp1) {
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
// STRICT-FP-LABEL: @sincos_f128
//      STRICT-FP:    call void @sincosl(
//
void sincos_f128(long double x, long double* ldp0, long double* ldp1) {
  sincosl(x, ldp0, ldp1);
}

// NO-MATH-ERRNO-LABEL: @sincos_builtin_f128
//      NO-MATH-ERRNO:    [[SINCOS:%.*]] = tail call { fp128, fp128 } @llvm.sincos.f128(fp128 {{.*}})
// NO-MATH-ERRNO-NEXT:    [[SIN:%.*]] = extractvalue { fp128, fp128 } [[SINCOS]], 0
// NO-MATH-ERRNO-NEXT:    [[COS:%.*]] = extractvalue { fp128, fp128 } [[SINCOS]], 1
// NO-MATH-ERRNO-NEXT:    store fp128 [[SIN]], ptr {{.*}}, align 16, !alias.scope [[SINCOS_ALIAS_SCOPE:![0-9]+]]
// NO-MATH-ERRNO-NEXT:    store fp128 [[COS]], ptr {{.*}}, align 16, !noalias [[SINCOS_ALIAS_SCOPE]]
//
// MATH-ERRNO-LABEL: @sincos_builtin_f128
//      MATH-ERRNO:    call void @sincosl(
//
void sincos_builtin_f128(long double x, long double* ldp0, long double* ldp1) {
  __builtin_sincosl(x, ldp0, ldp1);
}

// NO-MATH-ERRNO-LABEL: @sincospi_f32
//      NO-MATH-ERRNO:    [[SINCOSPI:%.*]] = tail call { float, float } @llvm.sincospi.f32(float {{.*}})
// NO-MATH-ERRNO-NEXT:    [[SINPI:%.*]] = extractvalue { float, float } [[SINCOSPI]], 0
// NO-MATH-ERRNO-NEXT:    [[COSPI:%.*]] = extractvalue { float, float } [[SINCOSPI]], 1
// NO-MATH-ERRNO-NEXT:    store float [[SINPI]], ptr {{.*}}, align 4, !alias.scope [[SINCOSPI_ALIAS_SCOPE:![0-9]+]]
// NO-MATH-ERRNO-NEXT:    store float [[COSPI]], ptr {{.*}}, align 4, !noalias [[SINCOSPI_ALIAS_SCOPE]]
//
// MATH-ERRNO-LABEL: @sincospi_f32
//      MATH-ERRNO:    call void @sincospif(
//
// STRICT-FP-LABEL: @sincospi_f32
//      STRICT-FP:    call void @sincospif(
//
void sincospi_f32(float x, float* fp0, float* fp1) {
  __builtin_sincospif(x, fp0, fp1);
}

// NO-MATH-ERRNO-LABEL: @sincospi_f64
//      NO-MATH-ERRNO:    [[SINCOSPI:%.*]] = tail call { double, double } @llvm.sincospi.f64(double {{.*}})
// NO-MATH-ERRNO-NEXT:    [[SINPI:%.*]] = extractvalue { double, double } [[SINCOSPI]], 0
// NO-MATH-ERRNO-NEXT:    [[COSPI:%.*]] = extractvalue { double, double } [[SINCOSPI]], 1
// NO-MATH-ERRNO-NEXT:    store double [[SINPI]], ptr {{.*}}, align 8, !alias.scope [[SINCOSPI_ALIAS_SCOPE:![0-9]+]]
// NO-MATH-ERRNO-NEXT:    store double [[COSPI]], ptr {{.*}}, align 8, !noalias [[SINCOSPI_ALIAS_SCOPE]]
//
// MATH-ERRNO-LABEL: @sincospi_f64
//      MATH-ERRNO:    call void @sincospi(
//
// STRICT-FP-LABEL: @sincospi_f64
//      STRICT-FP:    call void @sincospi(
//
void sincospi_f64(double x, double* dp0, double* dp1) {
  __builtin_sincospi(x, dp0, dp1);
}

// NO-MATH-ERRNO-LABEL: @sincospi_f128
//      NO-MATH-ERRNO:    [[SINCOSPI:%.*]] = tail call { fp128, fp128 } @llvm.sincospi.f128(fp128 {{.*}})
// NO-MATH-ERRNO-NEXT:    [[SINPI:%.*]] = extractvalue { fp128, fp128 } [[SINCOSPI]], 0
// NO-MATH-ERRNO-NEXT:    [[COSPI:%.*]] = extractvalue { fp128, fp128 } [[SINCOSPI]], 1
// NO-MATH-ERRNO-NEXT:    store fp128 [[SINPI]], ptr {{.*}}, align 16, !alias.scope [[SINCOSPI_ALIAS_SCOPE:![0-9]+]]
// NO-MATH-ERRNO-NEXT:    store fp128 [[COSPI]], ptr {{.*}}, align 16, !noalias [[SINCOSPI_ALIAS_SCOPE]]
//
// MATH-ERRNO-LABEL: @sincospi_f128
//      MATH-ERRNO:    call void @sincospil(
//
// STRICT-FP-LABEL: @sincospi_f128
//      STRICT-FP:    call void @sincospil(
//
void sincospi_f128(long double x, long double* ldp0, long double* ldp1) {
  __builtin_sincospil(x, ldp0, ldp1);
}
