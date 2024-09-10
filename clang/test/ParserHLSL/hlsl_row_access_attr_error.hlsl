// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -o - %s -verify

// expected-error@+1{{'row_access' attribute cannot be applied to a declaration}}
[[hlsl::row_access]] __hlsl_resource_t res0;

// expected-error@+1{{'row_access' attribute takes no arguments}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::row_access(3)]] res2;
  
// expected-error@+1{{use of undeclared identifier 'gibberish'}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::row_access(gibberish)]] res3;

// expected-warning@+1{{attribute 'row_access' is already applied}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::row_access]] [[hlsl::row_access]] res4;
