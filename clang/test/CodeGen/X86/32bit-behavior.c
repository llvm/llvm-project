// SSE
// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature +sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=source \
// RUN: | FileCheck -check-prefix=CHECK-SRC %s

// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature +sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=double \
// RUN: | FileCheck -check-prefix=CHECK-DBL %s

// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature +sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=extended \
// RUN: | FileCheck -check-prefix=CHECK-DBL %s

// SSE Fast Math
// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature +sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=source \
// RUN: -ffast-math | FileCheck -check-prefix=CHECK-FM-SRC %s

// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature +sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=double \
// RUN: -ffast-math | FileCheck -check-prefix=CHECK-FM %s

// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature +sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=extended \
// RUN: -ffast-math | FileCheck -check-prefix=CHECK-FM %s

// NO SSE
// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature -sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=source  \
// RUN: | FileCheck -check-prefix=CHECK-SRC %s

// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature -sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=double \
// RUN: | FileCheck -check-prefix=CHECK-DBL %s

// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature -sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=extended \
// RUN: | FileCheck -check-prefix=CHECK-DBL %s

// NO SSE Fast Math
// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature -sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=source \
// RUN: -ffast-math | FileCheck -check-prefix=CHECK %s

// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature -sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=double \
// RUN: -ffast-math | FileCheck -check-prefix=CHECK-DBL-FM %s

// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature -sse \
// RUN: -emit-llvm -o - %s -ffp-eval-method=extended \
// RUN: -ffast-math | FileCheck -check-prefix=CHECK-DBL-FM %s

float addit(float a, float b, float c) {
  // CHECK-SRC: load float, float*
  // CHECK-SRC: load float, float*
  // CHECK-SRC: fadd float
  // CHECK-SRC: load float, float*
  // CHECK-SRC: fadd float

  // CHECK-FM-SRC: load float, float*
  // CHECK-FM-SRC: load float, float*
  // CHECK-FM-SRC: fadd reassoc nnan ninf nsz arcp afn float
  // CHECK-FM-SRC: load float, float*
  // CHECK-FM-SRC: fadd reassoc nnan ninf nsz arcp afn float

  // CHECK-FM: load float, float*
  // CHECK-FM: fpext float {{.*}} to double
  // CHECK-FM: load float, float*
  // CHECK-FM: fpext float {{.*}} to double
  // CHECK-FM: fadd reassoc nnan ninf nsz arcp afn double
  // CHECK-FM: load float, float*
  // CHECK-FM: fadd reassoc nnan ninf nsz arcp afn double
  // CHECK-FM: fptrunc double {{.*}} to float

  // CHECK-DBL: load float, float*
  // CHECK-DBL: fpext float {{.*}} to double
  // CHECK-DBL: load float, float*
  // CHECK-DBL: fpext float {{.*}} to double
  // CHECK-DBL: fadd double
  // CHECK-DBL: load float, float*
  // CHECK-DBL: fpext float {{.*}} to double
  // CHECK-DBL: fadd double
  // CHECK-DBL: fptrunc double {{.*}} to float

  // CHECK-DBL-FM: load float, float*
  // CHECK-DBL-FM: fpext float {{.*}} to double
  // CHECK-DBL-FM: load float, float*
  // CHECK-DBL-FM: fpext float {{.*}} to double
  // CHECK-DBL-FM: fadd reassoc nnan ninf nsz arcp afn double
  // CHECK-DBL-FM: load float, float*
  // CHECK-DBL-FM: fpext float {{.*}} to double
  // CHECK-DBL-FM: fadd reassoc nnan ninf nsz arcp afn double
  // CHECK-DBL-FM: fptrunc double {{.*}} to float

  // CHECK: ret float
  return a + b + c;
}
