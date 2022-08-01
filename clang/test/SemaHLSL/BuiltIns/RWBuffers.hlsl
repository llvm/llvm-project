// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify %s

Resource ResourceDescriptorHeap[5];
typedef vector<float, 3> float3;

RWBuffer<float3> Buffer;

[numthreads(1,1,1)]
void main() {
  (void)Buffer.h; // expected-error {{'h' is a private member of 'hlsl::RWBuffer<float __attribute__((ext_vector_type(3)))>'}}
  // expected-note@* {{implicitly declared private here}}
}
