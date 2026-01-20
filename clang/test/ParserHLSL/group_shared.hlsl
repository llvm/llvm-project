// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -std=hlsl202x -o - -fsyntax-only %s -verify
extern groupshared float f;
extern float groupshared f; // Ok, redeclaration?


// expected-warning@#gs_lambda {{lambdas are a clang HLSL extension}}
// expected-error@#gs_lambda {{expected body of lambda expression}}
// expected-warning@#gs_lambda {{'auto' type specifier is a HLSL 202y extension}}
auto l = []() groupshared  {}; // #gs_lambda

float groupshared [[]] i = 12;

float groupshared const i2 = 12;
