// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -std=hlsl2018 -verify -Wconversion %s

groupshared uint SharedData;

void fn1(groupshared uint Sh) {
// expected-warning@-1{{support for groupshared parameter annotation not added until HLSL 202x}}
// expected-warning@*{{support for HLSL language version hlsl2018 is incomplete, recommend using hlsl202x instead}}
  Sh = 5;
}

template<typename T>
void fnT(T A, T B) {
  A = B;
}

template void fnT<groupshared int>(groupshared int A, groupshared int B);
// expected-warning@-1{{support for groupshared parameter annotation not added until HLSL 202x}}
// expected-warning@-2{{support for groupshared parameter annotation not added until HLSL 202x}}

void fn2() {
  fn1(SharedData);
}
