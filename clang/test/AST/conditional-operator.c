// RUN: %clang_cc1 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -menable-no-infs -fapprox-func -funsafe-math-optimizations \
// RUN: -fno-signed-zeros -mreassociate -freciprocal-math -ffp-contract=fast -ffast-math %s | FileCheck %s
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -include-pch %t -ast-dump-all /dev/null | FileCheck %s

float test_precise_off(int c, float t, float f) {
#pragma float_control(precise, off)
  return c ? t : f;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_precise_off
// CHECK: ConditionalOperator {{.*}} FPContractMode=2 AllowFPReassociate=1 NoHonorNaNs=1 NoHonorInfs=1 NoSignedZero=1 AllowReciprocal=1 AllowApproxFunc=1 MathErrno=0

float test_precise_on(int c, float t, float f) {
#pragma float_control(precise, on)
  return c ? t : f;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_precise_on
// CHECK: ConditionalOperator {{.*}} FPContractMode=1 AllowFPReassociate=0 NoHonorNaNs=0 NoHonorInfs=0 NoSignedZero=0 AllowReciprocal=0 AllowApproxFunc=0 MathErrno=1
