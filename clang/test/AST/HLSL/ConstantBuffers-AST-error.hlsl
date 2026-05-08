// RUN: not %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump -finclude-default-header -o - %s 2>&1 | FileCheck %s

// Unimplemented: https://github.com/llvm/llvm-project/issues/195093
// Once fixed, these tests should work and we should check the AST.

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
