// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -std=hlsl202x  -o - -fsyntax-only %s -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -std=hlsl202y  -o - -fsyntax-only %s -verify

#if __HLSL_VERSION < 2029
// expected-warning@#func{{'auto' type specifier is a HLSL 202y extension}}
// expected-warning@#func_gs{{'auto' type specifier is a HLSL 202y extension}}
// expected-warning@#l{{'auto' type specifier is a HLSL 202y extension}}
// expected-warning@#l2{{'auto' type specifier is a HLSL 202y extension}}
#endif

// expected-error@#func {{return type cannot be qualified with address space}}
auto func() -> groupshared void; // #func

// expected-error@#func_gs {{parameter may not be qualified with an address space}}
auto func(float groupshared) -> void; // #func_gs


// expected-error@#l {{parameter may not be qualified with an address space}}
// expected-warning@#l {{lambdas are a clang HLSL extension}}
auto l = [](groupshared float ) {}; // #l

// expected-error@#l2 {{return type cannot be qualified with address space}}
// expected-warning@#l2 {{lambdas are a clang HLSL extension}}
auto l2 = []() -> groupshared void {}; // #l2

struct S {
// expected-error@+1 {{return type cannot be qualified with address space}}
operator groupshared int() const;

};
