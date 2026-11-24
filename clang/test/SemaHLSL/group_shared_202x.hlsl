// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -std=hlsl202x  -o - -fsyntax-only %s -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -std=hlsl202y  -o - -fsyntax-only %s -verify

// expected-error@#func {{return type cannot be qualified with address space}}
auto func() -> groupshared void; // #func

auto func(float groupshared) -> void; // #func_gs

// expected-warning@#l {{lambdas are a clang HLSL extension}}
auto l = [](groupshared float ) {}; // #l

// expected-error@#l2 {{return type cannot be qualified with address space}}
// expected-warning@#l2 {{lambdas are a clang HLSL extension}}
auto l2 = []() -> groupshared void {}; // #l2

struct S {
// expected-error@+1 {{return type cannot be qualified with address space}}
operator groupshared int() const;

};
