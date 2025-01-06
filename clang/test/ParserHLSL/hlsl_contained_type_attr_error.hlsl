// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -std=hlsl202x -x hlsl -o - %s -verify

typedef vector<float, 4> float4;

// expected-error@+1{{'contained_type' attribute cannot be applied to a declaration}}
[[hlsl::contained_type(float4)]] __hlsl_resource_t h1;

// expected-error@+1{{'contained_type' attribute takes one argument}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type()]] h3;

// expected-error@+1{{expected a type}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(0)]] h4;

// expected-error@+1{{unknown type name 'a'}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(a)]] h5;

// expected-error@+1{{expected a type}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type("b", c)]] h6;

// expected-warning@+1{{attribute 'contained_type' is already applied}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(float)]] [[hlsl::contained_type(float)]] h7;

// expected-warning@+1{{attribute 'contained_type' is already applied with different arguments}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(float)]] [[hlsl::contained_type(int)]] h8;

// expected-error@+2{{attribute 'resource_class' can be used only on HLSL intangible type '__hlsl_resource_t'}}
// expected-error@+1{{attribute 'contained_type' can be used only on HLSL intangible type '__hlsl_resource_t'}}
float [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(float)]] res5;
