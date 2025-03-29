// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -std=hlsl202x  -o - -fsyntax-only %s -verify
extern groupshared float f;
extern float groupshared f; // Ok, redeclaration?

// expected-error@#l {{return type cannot be qualified with address space}}
// expected-warning@#l {{lambdas are a clang HLSL extension}}
// expected-warning@#l{{'auto' type specifier is a HLSL 202y extension}}
auto l = []() -> groupshared void {}; // #l
// expected-error@#l2 {{expected a type}}
// expected-warning@#l2 {{lambdas are a clang HLSL extension}}
// expected-warning@#l2{{'auto' type specifier is a HLSL 202y extension}}
auto l2 = []() -> groupshared {}; // #l2

float groupshared [[]] i = 12;

float groupshared const i2 = 12;

void foo() {
    l();
}

extern groupshared float f;
const float cf = f;
// expected-error@#func{{'auto' return without trailing return type; deduced return types are a C++14 extension}}
// expected-warning@#func{{'auto' type specifier is a HLSL 202y extension}}
auto func() { // #func
  return f;
}

void other() {
  // NOTE: groupshared and const are stripped off thanks to lvalue to rvalue
  // conversions and we deduce float for the return type.
  // expected-warning@#local{{lambdas are a clang HLSL extension}}
  // expected-warning@#local{{'auto' type specifier is a HLSL 202y extension}}
  auto l = [&]() { return f; }; // #local
  // expected-warning@#local2{{lambdas are a clang HLSL extension}}
  // expected-warning@#local2{{'auto' type specifier is a HLSL 202y extension}}
  auto l2 = [&]() { return cf; }; // #local2
}
