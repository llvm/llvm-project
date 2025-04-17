// RUN: %clang_cc1 -Wno-error=hlsl-implicit-binding -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// expected-warning@+1 {{resource has implicit register binding}}
cbuffer cb0 {
  int a;
}

// No warning - this is an element of the $Globals buffer not it's own binding.
float b;

// expected-warning@+1 {{resource has implicit register binding}}
RWBuffer<int> c;

// No warning - explicit binding.
RWBuffer<float> d : register(u0);

// TODO: Add this test once #135287 lands
// TODO: ... @+1 {{resource has implicit register binding}}
// TODO: RWBuffer<float> dd : register(space1);

// No warning - explicit binding.
RWBuffer<float> ddd : register(u3, space4);

struct S { int x; };
// expected-warning@+1 {{resource has implicit register binding}}
StructuredBuffer<S> e;

// No warning - __hlsl_resource_t isn't itself a resource object.
__hlsl_resource_t [[hlsl::resource_class(SRV)]] f;

struct CustomSRV {
  __hlsl_resource_t [[hlsl::resource_class(SRV)]] x;
};
// expected-warning@+1 {{resource has implicit register binding}}
CustomSRV g;

// expected-warning@+1 {{resource has implicit register binding}}
RWBuffer<float> h[10];

// No warning - explicit binding.
RWBuffer<float> hh[100] : register(u4);
