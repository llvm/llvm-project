// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -o - %s -verify

struct SomeType {
  int i;  
};

// expected-error@+1{{'resource_class' attribute cannot be applied to a declaration}}
[[hlsl::resource_class(UAV)]] SomeType e0;

// expected-error@+1{{'resource_class' attribute takes one argument}}
SomeType [[hlsl::resource_class()]] e1;

// expected-warning@+1{{ResourceClass attribute argument not supported: gibberish}}
SomeType [[hlsl::resource_class(gibberish)]] e2;

// expected-warning@+1{{attribute 'resource_class' is already applied with different arguments}}
SomeType [[hlsl::resource_class(SRV)]] [[hlsl::resource_class(UAV)]] e3;

// duplicate attribute with the same meaning - no error
SomeType [[hlsl::resource_class(SRV)]] [[hlsl::resource_class(SRV)]] e4;

// expected-error@+1{{'resource_class' attribute takes one argument}}
SomeType [[hlsl::resource_class(SRV, "aa")]] e5;
