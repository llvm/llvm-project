// RUN: not %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

// Unimplemented: https://github.com/llvm/llvm-project/issues/195093
// These cases should work. When fixed we should add proper CHECKs.

struct S {
  float3 a;
  float2 b;
};

cbuffer CB {
  S s_cb;
}

ConstantBuffer<S> cb;

[numthreads(1,1,1)]
void main() {
  // CHECK: error: no matching constructor for initialization of 'S'
  S l1 = s_cb;

  // CHECK: error: no viable constructor copying variable of type 'const hlsl_constant S'
  S l2 = cb;
}
