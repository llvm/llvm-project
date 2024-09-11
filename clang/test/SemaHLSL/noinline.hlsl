// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm-only %s -verify

RWBuffer<float> buf;

__attribute__((noinline))
float callMeMaybe() {// expected-warning {{ignoring the 'noinline' attribute because HLSL does not support it}}
  return 0.0;
}

[numthreads(1,1,1)]
void main() {
     buf[0] = callMeMaybe();     
}
