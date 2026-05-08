// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -finclude-default-header -fsyntax-only -verify %s

struct S { // expected-note 3 {{candidate constructor}}
  float a;
  int b;
};

struct Empty {};

struct ContainsResource {
  Texture2D tex;
};

union U {
  float a;
  int b;
};

// Valid
ConstantBuffer<S> cb;
ConstantBuffer<Empty> cb_empty;

// Invalid: non-struct/class
// expected-error@+1 {{constraints not satisfied for class template 'ConstantBuffer'}}
ConstantBuffer<float> cb_float;
// expected-note@* {{because 'float' does not satisfy '__is_constant_buffer_element_compatible'}}
// expected-note@* {{because '__builtin_hlsl_is_constant_buffer_element_compatible(float)' evaluated to false}}

// expected-error@+1 {{constraints not satisfied for class template 'ConstantBuffer'}}
ConstantBuffer<float4> cb_float4;
// expected-note@* {{because 'float4' (aka 'vector<float, 4>') does not satisfy '__is_constant_buffer_element_compatible'}}
// expected-note@* {{because '__builtin_hlsl_is_constant_buffer_element_compatible(vector<float, 4>)' evaluated to false}}

// expected-error@+1 {{constraints not satisfied for class template 'ConstantBuffer'}}
ConstantBuffer<float[4]> cb_array;
// expected-note@* {{because 'float[4]' does not satisfy '__is_constant_buffer_element_compatible'}}
// expected-note@* {{because '__builtin_hlsl_is_constant_buffer_element_compatible(float[4])' evaluated to false}}

// Invalid: contains resource
// expected-error@+1 {{constraints not satisfied for class template 'ConstantBuffer'}}
ConstantBuffer<ContainsResource> cb_res;
// expected-note@* {{because 'ContainsResource' does not satisfy '__is_constant_buffer_element_compatible'}}
// expected-note@* {{because '__builtin_hlsl_is_constant_buffer_element_compatible(ContainsResource)' evaluated to false}}

// Invalid: intangible type
// expected-error@+1 {{use of class template 'Texture2D' requires template arguments}}
ConstantBuffer<Texture2D> cb_tex;
// expected-note@* {{template declaration from hidden source}}

// Invalid: intangible type
// expected-error@+1 {{constraints not satisfied for class template 'ConstantBuffer'}}
ConstantBuffer<Texture2D<float>> cb_tex;
// expected-note@* {{because 'Texture2D<float>' does not satisfy '__is_constant_buffer_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_constant_buffer_element_compatible(hlsl::Texture2D<float>)' evaluated to false}}

// Invalid: union
// expected-error@+1 {{constraints not satisfied for class template 'ConstantBuffer'}}
ConstantBuffer<U> cb_union;
// expected-note@* {{because 'U' does not satisfy '__is_constant_buffer_element_compatible'}}
// expected-note@* {{because '__builtin_hlsl_is_constant_buffer_element_compatible(U)' evaluated to false}}

void takes_inout_s(inout S s) {}

void foo() {
  // This case should fail because we cannot writeback to `cb` after the call.
  // expected-error@+1 {{no viable constructor copying parameter of type 'const hlsl_constant S'}}
  takes_inout_s(cb);
}

void test_direct_assignment() {
  // expected-error@+2 {{cannot assign to return value because function 'operator const hlsl_constant S &' returns a const value}}
  // expected-note@* {{function 'operator const hlsl_constant S &' which returns const-qualified type 'const hlsl_constant S &' declared here}}
  cb.a = 5.0;
}
