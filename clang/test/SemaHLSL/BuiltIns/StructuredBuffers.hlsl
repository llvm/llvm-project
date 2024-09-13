// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify %s

typedef vector<float, 3> float3;

StructuredBuffer<float3> Buffer;

// expected-error@+2 {{class template 'StructuredBuffer' requires template arguments}}
// expected-note@*:* {{template declaration from hidden source: template <class element_type> class StructuredBuffer}}
StructuredBuffer BufferErr1;

// expected-error@+2 {{too few template arguments for class template 'StructuredBuffer'}}
// expected-note@*:* {{template declaration from hidden source: template <class element_type> class StructuredBuffer}}
StructuredBuffer<> BufferErr2;

[numthreads(1,1,1)]
void main() {
  (void)Buffer.h; // expected-error {{'h' is a private member of 'hlsl::StructuredBuffer<vector<float, 3> >'}}
  // expected-note@* {{implicitly declared private here}}
}
