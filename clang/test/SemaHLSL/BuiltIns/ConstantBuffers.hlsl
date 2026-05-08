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
