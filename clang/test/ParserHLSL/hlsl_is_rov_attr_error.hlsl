// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s -verify

// expected-error@+1{{'is_rov' attribute takes no arguments}}
struct [[hlsl::is_rov(3)]] Eg1 {
  int i;  
};

Eg1 e1;

// expected-error@+1{{use of undeclared identifier 'gibberish'}}
struct [[hlsl::is_rov(gibberish)]] Eg2 {
  int i;  
};

Eg2 e2;
