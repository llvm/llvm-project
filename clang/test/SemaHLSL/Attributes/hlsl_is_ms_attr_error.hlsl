// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -o - %s -verify

// expected-error@+1{{'hlsl::is_ms' attribute cannot be applied to a declaration}}
[[hlsl::is_ms]] __hlsl_resource_t res0;

// expected-error@+1{{HLSL resource needs to have [[hlsl::resource_class()]] attribute}}
__hlsl_resource_t [[hlsl::is_ms]] res1;

// expected-error@+1{{'hlsl::is_ms' attribute takes no arguments}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::is_ms(3)]] res2;

// expected-error@+1{{use of undeclared identifier 'gibberish'}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::is_ms(gibberish)]] res3;

// expected-warning@+1{{attribute 'hlsl::is_ms' is already applied}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::is_ms]] [[hlsl::is_ms]] res4;

// expected-error@+2{{attribute 'hlsl::resource_class' can be used only on HLSL intangible type '__hlsl_resource_t'}}
// expected-error@+1{{attribute 'hlsl::is_ms' can be used only on HLSL intangible type '__hlsl_resource_t'}}
float [[hlsl::resource_class(UAV)]] [[hlsl::is_ms]] res5;
