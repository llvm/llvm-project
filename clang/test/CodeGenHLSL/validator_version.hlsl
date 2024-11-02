// RUN: %clang -cc1 -S -triple dxil-pc-shadermodel6.3-library -S -emit-llvm -xhlsl -validator-version 1.1 -o - %s | FileCheck %s

// CHECK:!"dx.valver", ![[valver:[0-9]+]]}
// CHECK:![[valver]] = !{i32 1, i32 1}

float bar(float a, float b);

float foo(float a, float b) {
  return bar(a, b);
}
