// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s -verify

// expected-error@+1{{'texture_dimension' attribute takes one argument}}
struct [[hlsl::texture_dimension(3, 2)]] Eg1 {
  int i;  
};

Eg1 e1;

// expected-error@+1{{'texture_dimension' attribute takes one argument}}
struct [[hlsl::texture_dimension]] Eg2 {
  int i;  
};

Eg2 e2;

// expected-error@+1{{use of undeclared identifier 'gibberish'}}
struct [[hlsl::texture_dimension(gibberish)]] Eg3 {
  int i;  
};

Eg2 e3;
