// SSE
// RUN: %clang_cc1  \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature +sse \
// RUN: -emit-llvm -o - %s | FileCheck -check-prefix=CHECK %s

// NO SSE
// RUN: %clang_cc1  \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature -sse \
// RUN: -emit-llvm -o - %s | FileCheck -check-prefix=CHECK %s

// NO SSE Fast Math
// RUN: %clang_cc1  \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature -sse \
// RUN: -ffast-math -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-FM %s

float addit(float a, float b, float c) {
  // CHECK: load float, ptr
  // CHECK: load float, ptr
  // CHECK: fadd float
  // CHECK: load float, ptr
  // CHECK: fadd float

  // CHECK-FM: load float, ptr
  // CHECK-FM: load float, ptr
  // CHECK-FM: fadd reassoc nnan ninf nsz arcp afn float
  // CHECK-FM: load float, ptr
  // CHECK-FM: fadd reassoc nnan ninf nsz arcp afn float

  return a + b + c;
}
