// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -o - %s -verify

// expected-error@+1{{'is_rov' attribute cannot be applied to a declaration}}
[[hlsl::is_rov]] __hlsl_resource_t res0;

// expected-error@+1{{HLSL resource needs to have [[hlsl::resource_class()]] attribute}}
__hlsl_resource_t [[hlsl::is_rov]] res1;

// expected-error@+1{{'is_rov' attribute takes no arguments}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::is_rov(3)]] res2;
  
// expected-error@+1{{use of undeclared identifier 'gibberish'}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::is_rov(gibberish)]] res3;

// expected-warning@+1{{attribute 'is_rov' is already applied}}
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::is_rov]] [[hlsl::is_rov]] res4;

// expected-error@+2{{attribute 'resource_class' can be used only on HLSL intangible type '__hlsl_resource_t'}}
// expected-error@+1{{attribute 'is_rov' can be used only on HLSL intangible type '__hlsl_resource_t'}}
float [[hlsl::resource_class(UAV)]] [[hlsl::is_rov]] res5;
