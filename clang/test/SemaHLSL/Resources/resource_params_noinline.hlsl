// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -fsyntax-only -finclude-default-header %s -verify

// expected-error@+2 {{resource type 'RWBuffer<float>' cannot be used as a parameter of a noinline function}}
// expected-note@+1 {{attribute is here}}
__attribute__((noinline)) float resource_param(RWBuffer<float> A) {}

// expected-error@+2 {{resource type 'RWBuffer<float>[4]' cannot be used as a parameter of a noinline function}}
// expected-note@+1 {{attribute is here}}
__attribute__((noinline)) float resource_array_param(RWBuffer<float> A[4]) {}

struct MyStruct {
  RWBuffer<float> A;
};
// expected-error@+2 {{resource type 'MyStruct' cannot be used as a parameter of a noinline function}}
// expected-note@+1 {{attribute is here}}
__attribute__((noinline)) float resource_struct_param(MyStruct S) {}

// expected-error@+2 {{resource type 'RWByteAddressBuffer' cannot be used as a parameter of a noinline function}}
// expected-note@+1 {{attribute is here}}
__attribute__((noinline)) void no_prior_use(RWByteAddressBuffer A) {}
