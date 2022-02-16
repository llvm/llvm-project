// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -target-feature -sse -emit-llvm -o - -verify %s
//
// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -emit-llvm -o - -verify=no-warn %s

// no-warn-no-diagnostics

float add2(float a, float b, float c) {
#pragma clang fp eval_method(source)
  return a + b + c;
} // expected-warning{{Setting FPEvalMethod to source on a 32bit target, with no SSE is not supported.}}
