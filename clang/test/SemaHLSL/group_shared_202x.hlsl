// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -std=hlsl202x  -o - -fsyntax-only %s -verify

// expected-error@+1 {{return type cannot be qualified with address space}}
auto func() -> groupshared void;

// expected-error@+1 {{parameter may not be qualified with an address space}}
auto func(float groupshared) -> void;

// expected-error@+1 {{parameter may not be qualified with an address space}}
auto l = [](groupshared float ) {};

// expected-error@+1 {{return type cannot be qualified with address space}}
auto l2 = []() -> groupshared void {};

struct S {
// expected-error@+1 {{return type cannot be qualified with address space}}
operator groupshared int() const;

};
