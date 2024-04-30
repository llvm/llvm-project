// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify
extern groupshared float f;
extern float groupshared f; // Ok, redeclaration?


// expected-warning@+3 {{lambdas are a C++11 extension}}
// expected-error@+2   {{expected body of lambda expression}}
// expected-warning@+1 {{'auto' type specifier is a C++11 extension}}
auto l = []() groupshared  {};

float groupshared [[]] i = 12;

float groupshared const i2 = 12;
