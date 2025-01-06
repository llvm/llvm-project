// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -xhlsl -o - %s | FileCheck %s

// CHECK:!dx.valver = !{![[valver:[0-9]+]]}
// CHECK:![[valver]] = !{i32 1, i32 8}

float bar(float a, float b);

float foo(float a, float b) {
  return bar(a, b);
}
