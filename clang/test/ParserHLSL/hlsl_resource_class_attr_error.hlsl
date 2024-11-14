// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s -verify

struct Eg1 {
// expected-error@+1{{'resource_class' attribute takes one argument}}
  [[hlsl::resource_class()]] int i;  
};

Eg1 e1;

struct Eg2 {
// expected-warning@+1{{ResourceClass attribute argument not supported: gibberish}}
  [[hlsl::resource_class(gibberish)]] int i;  
};

Eg2 e2;

// expected-warning@+1{{'resource_class' attribute only applies to non-static data members}}
struct [[hlsl::resource_class(SRV)]] Eg3 {
  int i;  
};

Eg3 e3;
