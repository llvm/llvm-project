// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -xhlsl -validator-version 1.1 -o - %s | FileCheck %s

// FIXME:The following line should work once SPIR-V support for HLSL is added.
// See: https://github.com/llvm/llvm-project/issues/57877
// DISABLED: %clang_cc1 -triple spirv32 -emit-llvm -xhlsl -validator-version 1.1 -o - %s | FileCheck %s --check-prefix=NOT_DXIL

// CHECK:!dx.valver = !{![[valver:[0-9]+]]}
// CHECK:![[valver]] = !{i32 1, i32 1}

// NOT_DXIL-NOT:!dx.valver

float bar(float a, float b);

float foo(float a, float b) {
  return bar(a, b);
}
