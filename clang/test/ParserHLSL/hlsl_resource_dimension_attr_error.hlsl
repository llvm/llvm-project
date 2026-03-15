// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -o - %s -verify

// expected-error@+1{{'hlsl::dimension' attribute cannot be applied to a declaration}}
[[hlsl::dimension("2D")]] __hlsl_resource_t e0;

// expected-error@+1{{'hlsl::dimension' attribute takes one argument}}
__hlsl_resource_t [[hlsl::dimension()]] e1;

// expected-error@+1{{expected string literal as argument of 'dimension' attribute}}
__hlsl_resource_t [[hlsl::dimension(2)]] e2;

// expected-warning@+1{{ResourceDimension attribute argument not supported: gibberish}}
__hlsl_resource_t [[hlsl::dimension("gibberish")]] e3;

// expected-error@+1{{'hlsl::dimension' attribute takes one argument}}
__hlsl_resource_t [[hlsl::dimension("2D", "3D")]] e4;

// expected-error@+1{{attribute 'hlsl::dimension' can be used only on HLSL intangible type '__hlsl_resource_t'}}
float [[hlsl::dimension("2D")]] e5;
