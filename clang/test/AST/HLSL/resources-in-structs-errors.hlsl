// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -verify %s

struct A {
  RWBuffer<float> Buf;
};

A incompleteArray[]; // expected-error {{definition of variable with array type needs an explicit size or an initializer}}
