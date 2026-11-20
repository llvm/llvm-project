// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm \
// RUN:   -debug-info-kind=constructor -fdx-no-source-metadata -o - \
// RUN:   -disable-llvm-passes %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm   -debug-info-kind=constructor -fdx-no-source-metadata -o -   -disable-llvm-passes %s | FileCheck %s %}

// CHECK-NOT: !dx.source.contents
// CHECK-NOT: !dx.source.defines
// CHECK-NOT: !dx.source.mainFileName
// CHECK-NOT: !dx.source.args

float foo(float a, float b) {
  return a + b;
}
