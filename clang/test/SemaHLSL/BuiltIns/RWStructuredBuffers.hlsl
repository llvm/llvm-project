// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify %s

typedef vector<float, 3> float3;

RWStructuredBuffer<float3> Buff;

// expected-error@+2 {{class template 'RWStructuredBuffer' requires template arguments}}
// expected-note@*:* {{template declaration from hidden source: template <typename element_type> requires __is_structured_resource_element_compatible<element_type> class RWStructuredBuffer {}}}
RWStructuredBuffer BufferErr1;

// expected-error@+2 {{too few template arguments for class template 'RWStructuredBuffer'}}
// expected-note@*:* {{template declaration from hidden source: template <typename element_type> requires __is_structured_resource_element_compatible<element_type> class RWStructuredBuffer {}}}
RWStructuredBuffer<> BufferErr2;

// test elements of 0 size
// expected-error@+3{{constraints not satisfied for class template 'RWStructuredBuffer' [with element_type = int[0]]}}
// expected-note@*:*{{because 'int[0]' does not satisfy '__is_structured_resource_element_compatible'}}
// expected-note@*:*{{because 'sizeof(int[0]) >= 1UL' (0 >= 1) evaluated to false}}
RWStructuredBuffer<int[0]> BufferErr3;

// In C++, empty structs do have a size of 1. So should HLSL.
// The concept will accept empty structs as element types, despite it being unintuitive.
struct Empty {};
RWStructuredBuffer<Empty> BufferErr4;

[numthreads(1,1,1)]
void main() {
  (void)Buff.__handle; // expected-error {{'__handle' is a private member of 'hlsl::RWStructuredBuffer<vector<float, 3>>'}}
  // expected-note@* {{implicitly declared private here}}
}

// expected-error@+2 {{class template 'RWStructuredBuffer' requires template arguments}}
// expected-note@*:* {{template declaration from hidden source: template <typename element_type> requires __is_structured_resource_element_compatible<element_type> class RWStructuredBuffer {}}}
void f1(RWStructuredBuffer B) {}

// expected-error@+2 {{class template 'RWStructuredBuffer' requires template arguments}}
// expected-note@*:* {{template declaration from hidden source: template <typename element_type> requires __is_structured_resource_element_compatible<element_type> class RWStructuredBuffer {}}}
RWStructuredBuffer f2();

struct S {
// expected-error@+2 {{class template 'RWStructuredBuffer' requires template arguments}}
// expected-note@*:* {{template declaration from hidden source: template <typename element_type> requires __is_structured_resource_element_compatible<element_type> class RWStructuredBuffer {}}}
  RWStructuredBuffer B;
};
