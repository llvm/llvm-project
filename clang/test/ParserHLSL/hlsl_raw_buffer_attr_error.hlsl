// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -o - %s -verify

// expected-error@+1{{'raw_buffer' attribute cannot be applied to a declaration}}
[[hlsl::raw_buffer]] __hlsl_resource_t res0;

// expected-error@+1{{'raw_buffer' attribute takes no arguments}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::raw_buffer(3)]] res2;
  
// expected-error@+1{{use of undeclared identifier 'gibberish'}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::raw_buffer(gibberish)]] res3;

// expected-warning@+1{{attribute 'raw_buffer' is already applied}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::raw_buffer]] [[hlsl::raw_buffer]] res4;
