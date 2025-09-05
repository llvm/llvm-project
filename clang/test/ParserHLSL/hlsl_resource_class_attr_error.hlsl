// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -o - %s -verify

// expected-error@+1{{'hlsl::resource_class' attribute cannot be applied to a declaration}}
[[hlsl::resource_class(UAV)]] __hlsl_resource_t e0;

// expected-error@+1{{'hlsl::resource_class' attribute takes one argument}}
__hlsl_resource_t [[hlsl::resource_class()]] e1;

// expected-warning@+1{{ResourceClass attribute argument not supported: gibberish}}
__hlsl_resource_t [[hlsl::resource_class(gibberish)]] e2;

// expected-warning@+1{{attribute 'hlsl::resource_class' is already applied with different arguments}}
__hlsl_resource_t [[hlsl::resource_class(SRV)]] [[hlsl::resource_class(UAV)]] e3;

// expected-warning@+1{{attribute 'hlsl::resource_class' is already applied}}
__hlsl_resource_t [[hlsl::resource_class(SRV)]] [[hlsl::resource_class(SRV)]] e4;

// expected-error@+1{{'hlsl::resource_class' attribute takes one argument}}
__hlsl_resource_t [[hlsl::resource_class(SRV, "aa")]] e5;

// expected-error@+1{{attribute 'hlsl::resource_class' can be used only on HLSL intangible type '__hlsl_resource_t'}}
float [[hlsl::resource_class(UAV)]] e6;
