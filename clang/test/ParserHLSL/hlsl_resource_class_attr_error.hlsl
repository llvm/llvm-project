// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s -verify

// expected-error@+1{{'resource_class' attribute takes one argument}}
struct [[hlsl::resource_class()]] Eg1 {
  int i;  
};

Eg1 e1;

// expected-warning@+1{{ResourceClass attribute argument not supported: gibberish}}
struct [[hlsl::resource_class(gibberish)]] Eg2 {
  int i;  
};

Eg2 e2;
