// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature -sse \
// RUN: -emit-llvm -ffp-eval-method=source  -o - -verify=no-warn %s
//
// ... unless -fexcess-precision=fast is set.
// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 -target-feature -sse \
// RUN: -target-feature +x87-excess-precision \
// RUN: -emit-llvm -ffp-eval-method=source  -o - -verify=warn %s
//
// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple i386-pc-windows -target-cpu pentium4 \
// RUN: -emit-llvm -ffp-eval-method=source  -o - -verify=no-warn %s

// no-warn-no-diagnostics

float add1(float a, float b, float c) {
  return a + b + c;
} // warn-warning{{setting the floating point evaluation method to `source` is not supported with '-fexcess-precision=fast' on a target without SSE}}

float add2(float a, float b, float c) {
#pragma clang fp eval_method(source)
  return a + b + c;
} // warn-warning{{setting the floating point evaluation method to `source` is not supported with '-fexcess-precision=fast' on a target without SSE}}
