// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -o - %s -verify

// expected-error@+1{{'is_rov' attribute cannot be applied to a declaration}}
[[hlsl::is_rov()]] __hlsl_resource_t res0;

// expected-error@+1{{HLSL resource needs to have [[hlsl::resource_class()]] attribute}}
__hlsl_resource_t [[hlsl::is_rov()]] res1;

// expected-error@+1{{'is_rov' attribute takes no arguments}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::is_rov(3)]] res2;
  
// expected-error@+1{{use of undeclared identifier 'gibberish'}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::is_rov(gibberish)]] res3;

// duplicate attribute with the same meaning - no error
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::is_rov()]] [[hlsl::is_rov()]] res4;
