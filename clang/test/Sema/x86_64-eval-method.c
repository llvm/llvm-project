// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -target-feature -sse -emit-llvm \
// RUN: -o - -verify=no-warn %s
//
// ... unless -fexcess-precision=fast is set.
// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -target-feature -sse \
// RUN: -target-feature +x87-excess-precision -emit-llvm -o - -verify=warn %s
//
// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -emit-llvm -o - -verify=no-warn %s

// no-warn-no-diagnostics

float add2(float a, float b, float c) {
#pragma clang fp eval_method(source)
  return a + b + c;
} // warn-warning{{setting the floating point evaluation method to `source` is not supported with '-fexcess-precision=fast' on a target without SSE}}
