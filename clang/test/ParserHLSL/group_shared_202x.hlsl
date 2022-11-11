// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -std=hlsl202x  -o - -fsyntax-only %s -verify
extern groupshared float f;
extern float groupshared f; // Ok, redeclaration?

// expected-error@+1 {{return type cannot be qualified with address space}}
auto l = []() -> groupshared void {};
// expected-error@+1 {{expected a type}}
auto l2 = []() -> groupshared {};

float groupshared [[]] i = 12;

float groupshared const i2 = 12;

void foo() {
    l();
}

extern groupshared float f;
const float cf = f;
// expected-error@+1 {{'auto' return without trailing return type; deduced return types are a C++14 extension}}
auto func() {
  return f;
}

void other() {
  // NOTE: groupshared and const are stripped off thanks to lvalue to rvalue conversions and we deduce float for the return type.
  auto l = [&]() { return f; };
  auto l2 = [&]() { return cf; };
}
