// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -verify -complex-range=promoted -o - | FileCheck %s

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -verify=nopromotion -complex-range=promoted -target-feature -x87 \
// RUN: -o - | FileCheck %s --check-prefix=NOX87

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-windows \
// RUN: -verify=nopromotion -complex-range=promoted -o - \
// RUN: | FileCheck %s --check-prefix=NOX87

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-windows \
// RUN: -verify=nopromotion -complex-range=promoted -target-feature -x87 \
// RUN: -o - | FileCheck %s --check-prefix=NOX87



// expected-no-diagnostics

// CHECK-LABEL: define dso_local <2 x float> @divd
_Complex float divd(_Complex float a, _Complex float b) {
  // CHECK: fpext float {{.*}} to double
  // CHECK: fpext float {{.*}} to double
  // CHECK: fdiv double
  // CHECK: fdiv double
  // CHECK: fptrunc double {{.*}} to float
  // CHECK: fptrunc double {{.*}} to float

  // NOX87: fpext float {{.*}} to double
  // NOX87: fpext float {{.*}} to double
  // NOX87: fdiv double
  // NOX87: fdiv double
  // NOX87: fptrunc double {{.*}} to float
  // NOX87: fptrunc double {{.*}} to float

  return a / b;
}

// CHECK-LABEL: define dso_local { double, double } @divf
_Complex double divf(_Complex double a, _Complex double b) {
  // CHECK: fpext double {{.*}} to x86_fp80
  // CHECK: fpext double {{.*}} to x86_fp80
  // CHECK: fdiv x86_fp80
  // CHECK: fdiv x86_fp80
  // CHECK: fptrunc x86_fp80
  // CHECK: fptrunc x86_fp80

  // NOX87: call double @llvm.fabs.f64(double {{.*}})
  // NOX87-NEXT: call double @llvm.fabs.f64(double {{.*}})
  // NOX87-NEXT: fcmp ugt double %{{.*}}, {{.*}}
  // NOX87-NEXT: br i1 {{.*}}, label
  // NOX87: abs_rhsr_greater_or_equal_abs_rhsi:
  // NOX87-NEXT: fdiv double
  // NOX87-NEXT: fmul double
  // NOX87-NEXT: fadd double
  // NOX87-NEXT: fmul double
  // NOX87-NEXT: fadd double
  // NOX87-NEXT: fdiv double
  // NOX87-NEXT: fmul double
  // NOX87-NEXT: fsub double
  // NOX87-NEXT: fdiv double
  // NOX87-NEXT: br label {{.*}}
  // NOX87: abs_rhsr_less_than_abs_rhsi:
  // NOX87-NEXT: fdiv double
  // NOX87-NEXT: fmul double
  // NOX87-NEXT: fadd double
  // NOX87-NEXT: fmul double
  // NOX87-NEXT: fadd double
  // NOX87-NEXT: fdiv double
  // NOX87-NEXT: fmul double
  // NOX87-NEXT: fsub double
  // NOX87-NEXT: fdiv double
  // NOX87-NEXT: br label
  // NOX87: complex_div:
  // NOX87-NEXT: phi double
  // NOX87-NEXT: phi double
  // NOX87-NEXT: getelementptr inbounds nuw { double, double }, ptr {{.*}}, i32 0, i32 0
  // NOX87-NEXT: getelementptr inbounds nuw { double, double }, ptr {{.*}}, i32 0, i32 1
  // NOX87-NEXT: store double
  // NOX87-NEXT: store double

  return a / b; // nopromotion-warning{{excess precision is requested but the target does not support excess precision which may result in observable differences in complex division behavior}}
}
