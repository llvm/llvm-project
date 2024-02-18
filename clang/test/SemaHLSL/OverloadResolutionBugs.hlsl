// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -o - -fsyntax-only %s -verify
// XFAIL: *

// https://github.com/llvm/llvm-project/issues/81047

// expected-no-diagnostics
void Fn3(double2 D);
void Fn3(float2 F);

void Call3(half2 H) {
  Fn3(H);
}

void Fn5(double2 D);

void Call5(half2 H) {
  Fn5(H);
}

void Fn4(int64_t2 L);
void Fn4(int2 I);

void Call4(int16_t H) {
  Fn4(H);
}

// https://github.com/llvm/llvm-project/issues/81049

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.2-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefix=NO_HALF

half sqrt_h(half x)
{
  return sqrt(x);
}

// NO_HALF: define noundef float @"?sqrt_h@@YA$halff@$halff@@Z"(
// NO_HALF: call float @llvm.sqrt.f32(float %0)
