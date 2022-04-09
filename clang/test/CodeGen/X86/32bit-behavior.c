// SSE
// RUN: %clang_cc1 -no-opaque-pointers -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature +sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=source \
// RUN: | FileCheck -check-prefix=CHECK-SRC %s

// RUN: %clang_cc1 -no-opaque-pointers -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature +sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=double \
// RUN: | FileCheck -check-prefix=CHECK-DBL %s

// RUN: %clang_cc1 -no-opaque-pointers -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature +sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=extended \
// RUN: | FileCheck -check-prefix=CHECK-DBL %s

// NO SSE
// RUN: %clang_cc1 -no-opaque-pointers -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature -sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=source  \
// RUN: | FileCheck -check-prefix=CHECK-SRC %s

// RUN: %clang_cc1 -no-opaque-pointers -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature -sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=double \
// RUN: | FileCheck -check-prefix=CHECK-DBL %s

// RUN: %clang_cc1 -no-opaque-pointers -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature -sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=extended \
// RUN: | FileCheck -check-prefix=CHECK-DBL %s

float addit(float a, float b, float c) {
  // CHECK-SRC: load float, float*
  // CHECK-SRC: load float, float*
  // CHECK-SRC: fadd float
  // CHECK-SRC: load float, float*
  // CHECK-SRC: fadd float

  // CHECK-DBL: load float, float*
  // CHECK-DBL: fpext float {{.*}} to double
  // CHECK-DBL: load float, float*
  // CHECK-DBL: fpext float {{.*}} to double
  // CHECK-DBL: fadd double
  // CHECK-DBL: load float, float*
  // CHECK-DBL: fpext float {{.*}} to double
  // CHECK-DBL: fadd double
  // CHECK-DBL: fptrunc double {{.*}} to float

  // CHECK: ret float
  return a + b + c;
}
