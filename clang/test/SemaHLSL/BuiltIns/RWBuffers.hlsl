// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify %s

typedef vector<float, 3> float3;

RWBuffer<float3> Buffer;

// expected-error@+2 {{class template 'RWBuffer' requires template arguments}}
// expected-note@*:* {{template declaration from hidden source: template <typename element_type> requires __is_typed_resource_element_compatible<element_type> class RWBuffer {}}}
RWBuffer BufferErr1;

// expected-error@+2 {{too few template arguments for class template 'RWBuffer'}}
// expected-note@*:* {{template declaration from hidden source: template <typename element_type> requires __is_typed_resource_element_compatible<element_type> class RWBuffer {}}}
RWBuffer<> BufferErr2;

struct threeDoubles {
  double a;
  double b;
  double c;
};

// expected-error@+3 {{constraints not satisfied for class template 'RWBuffer'}}
// expected-note@*:* {{because 'threeDoubles' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because 'sizeof(threeDoubles) <= 16UL' (24 <= 16) evaluated to false}}
RWBuffer<threeDoubles> BufferErr3;

[numthreads(1,1,1)]
void main() {
  (void)Buffer.h; // expected-error {{'h' is a private member of 'hlsl::RWBuffer<vector<float, 3>>'}}
  // expected-note@* {{implicitly declared private here}}
}
