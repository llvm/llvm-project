// RUN: not %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump -finclude-default-header -o - %s 2>&1 | FileCheck %s

// This test tracks issue #153055: Implicit conversion from hlsl_constant to generic address space is currently disabled.

struct S {
  float a;
};
ConstantBuffer<S> cb;

void takes_s(S s) {}

void main() {
  S s;

  // CHECK: error: no viable constructor copying parameter of type 'const hlsl_constant S'
  takes_s(cb);

  // CHECK: error: no viable constructor copying variable of type 'const hlsl_constant S'
  S s2 = cb;

  // CHECK: error: no viable conversion from 'ConstantBuffer<S>' to 'const S'
  s = cb;
}
