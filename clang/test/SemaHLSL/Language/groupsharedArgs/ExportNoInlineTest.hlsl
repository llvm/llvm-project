// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -std=hlsl202x -verify -Wconversion %s

export void fn1(groupshared uint Sh) {
// expected-error@-1{{'export' attribute is not compatible with 'groupshared' parameter attribute}}
  Sh = 5;
}

__attribute__((noinline)) void fn2(groupshared uint Sh) {
// expected-error@-1{{'noinline' attribute is not compatible with 'groupshared' parameter attribute}}
  Sh = 6;
}

template<typename T>
void fn3(T A, T B) {
  A = B;
}
