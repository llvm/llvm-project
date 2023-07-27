// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify
extern groupshared float f;
extern float groupshared f; // Ok, redeclaration?


// NOTE:lambda is not enabled except for hlsl202x.
// expected-error@+2 {{expected expression}}
// expected-warning@+1 {{'auto' type specifier is a C++11 extension}}
auto l = []() groupshared  {};

float groupshared [[]] i = 12;

float groupshared const i2 = 12;
