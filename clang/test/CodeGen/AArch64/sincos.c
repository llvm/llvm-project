// RUN: %clang_cc1 -triple=aarch64-gnu-linux -emit-llvm %s -o - | FileCheck --check-prefix=NO-MATH-ERRNO %s
// RUN: %clang_cc1 -triple=aarch64-gnu-linux -emit-llvm -fmath-errno %s -o - | FileCheck --check-prefix=MATH-ERRNO %s

void sincos(double, double*, double*);
void sincosf(float, float*, float*);

// NO-MATH-ERRNO-LABEL: @foo
//      NO-MATH-ERRNO:    [[SINCOS:%.*]] = call { double, double } @llvm.sincos.f64(double {{.*}})
// NO-MATH-ERRNO-NEXT:    [[SIN:%.*]] = extractvalue { double, double } [[SINCOS]], 0
// NO-MATH-ERRNO-NEXT:    [[COS:%.*]] = extractvalue { double, double } [[SINCOS]], 1
// NO-MATH-ERRNO-NEXT:    store double [[SIN]], ptr {{.*}}, align 8, !alias.scope [[SINCOS_ALIAS_SCOPE:![0-9]+]]
// NO-MATH-ERRNO-NEXT:    store double [[COS]], ptr {{.*}}, align 8, !noalias [[SINCOS_ALIAS_SCOPE]]
//
// MATH-ERRNO-LABEL: @foo
//      MATH-ERRNO:    call void @sincos(
//
void foo(double x, double* dp0, double* dp1) {
  sincos(x, dp0, dp1);
}

// NO-MATH-ERRNO-LABEL: @bar
//      NO-MATH-ERRNO:    [[SINCOS:%.*]] = call { float, float } @llvm.sincos.f32(float {{.*}})
// NO-MATH-ERRNO-NEXT:    [[SIN:%.*]] = extractvalue { float, float } [[SINCOS]], 0
// NO-MATH-ERRNO-NEXT:    [[COS:%.*]] = extractvalue { float, float } [[SINCOS]], 1
// NO-MATH-ERRNO-NEXT:    store float [[SIN]], ptr {{.*}}, align 4, !alias.scope [[SINCOS_ALIAS_SCOPE:![0-9]+]]
// NO-MATH-ERRNO-NEXT:    store float [[COS]], ptr {{.*}}, align 4, !noalias [[SINCOS_ALIAS_SCOPE]]
//
// MATH-ERRNO-LABEL: @bar
//      MATH-ERRNO:    call void @sincosf(
//
void bar(float x, float* fp0, float* fp1) {
  sincosf(x, fp0, fp1);
}
