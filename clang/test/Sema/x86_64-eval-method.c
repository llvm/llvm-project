// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -target-feature -sse -emit-llvm \
// RUN: -o - -verify=warn %s
//
// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -emit-llvm -o - -verify=no-warn %s

// no-warn-no-diagnostics

float add2(float a, float b, float c) {
#pragma clang fp eval_method(source)
  return a + b + c;
} // warn-warning{{Setting the floating point evaluation method to `source` on a target without SSE is not supported.}}
