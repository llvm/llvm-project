// RUN %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN -verify -complex-range=promoted -o - | FileCheck %s

// RUN %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN -verify=nopromotion -complex-range=promoted -target-feature -x87 -o - | FileCheck %s --check-prefix=X87

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-windows \
// RUN: -verify=nopromotion -complex-range=promoted -o - | FileCheck %s --check-prefix=X87

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-windows \
// RUN: -verify=nopromotion -complex-range=promoted -target-feature -x87 -o - | FileCheck %s --check-prefix=X87



// expected-no-diagnostics

// CHECK-LABEL: define dso_local <2 x float> @divd
_Complex float divd(_Complex float a, _Complex float b) {
  // CHECK: fpext float {{.*}} to double
  // CHECK: fpext float {{.*}} to double
  // CHECK: fdiv double
  // CHECK: fdiv double
  // CHECK: fptrunc double {{.*}} to float
  // CHECK: fptrunc double {{.*}} to float

  // X87: fpext float {{.*}} to double
  // X87: fpext float {{.*}} to double
  // X87: fdiv double
  // X87: fdiv double
  // X87: fptrunc double {{.*}} to float
  // X87: fptrunc double {{.*}} to float

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

  // X87: fdiv double
  // X87: fdiv double

  return a / b; // nopromotion-warning{{excess precision is requested but the target does not support excess precision which may result in observable differences in complex division behavior}}
}
