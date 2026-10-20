// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -std=hlsl202x -o - -fsyntax-only %s -verify
extern groupshared float f;
extern float groupshared f; // Ok, redeclaration?


// expected-warning@#gs_lambda {{lambdas are a clang HLSL extension}}
// expected-error@#gs_lambda {{expected body of lambda expression}}
// expected-warning@#gs_lambda {{'auto' type specifier is a HLSL 202y extension}}
auto l = []() groupshared  {}; // #gs_lambda

// expected-warning@#gs_no_parens {{lambdas are a clang HLSL extension}}
// expected-warning@#gs_no_parens {{'auto' type specifier is a HLSL 202y extension}}
// expected-error@#gs_no_parens {{expected body of lambda expression}}
auto l_no_parens = [] groupshared {}; // #gs_no_parens

float groupshared [[]] i = 12;
// expected-error@-1{{initializer of groupshared variable will be ignored}}

float groupshared const i2 = 12;
// expected-error@-1{{initializer of groupshared variable will be ignored}}
