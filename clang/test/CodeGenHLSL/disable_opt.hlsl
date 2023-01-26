// RUN: %clang_cc1 -S -triple dxil-pc-shadermodel6.3-library -O0 -emit-llvm -xhlsl  -o - %s | FileCheck %s
// RUN: %clang_cc1 -S -triple dxil-pc-shadermodel6.3-library -O3 -emit-llvm -xhlsl  -o - %s | FileCheck %s --check-prefix=OPT

// CHECK:!"dx.disable_optimizations", i32 1}

// OPT-NOT:"dx.disable_optimizations"

float bar(float a, float b);

float foo(float a, float b) {
  return bar(a, b);
}
