// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -std=hlsl2017 -o - -fsyntax-only %s -verify=expected,hlsl2017
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -std=hlsl2018 -o - -fsyntax-only %s -verify=expected,hlsl2018
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -std=hlsl2021 -o - -fsyntax-only %s -verify=expected,hlsl2021

// hlsl2017-warning@*{{support for HLSL language version hlsl2017 is incomplete, recommend using hlsl202x instead}}
// hlsl2018-warning@*{{support for HLSL language version hlsl2018 is incomplete, recommend using hlsl202x instead}}
// hlsl2021-warning@*{{support for HLSL language version hlsl2021 is incomplete, recommend using hlsl202x instead}}

// expected-warning@+1{{'auto'}}
auto foo(float A) -> float {
  return A;
}

void bar(float A) {
// expected-warning@+1{{'auto'}}
  auto C = A;
}
