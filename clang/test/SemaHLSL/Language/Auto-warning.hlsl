// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -std=hlsl2017 -o - -fsyntax-only %s -verify=expected,hlsl2017
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -std=hlsl2018 -o - -fsyntax-only %s -verify=expected,hlsl2018
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -std=hlsl2021 -o - -fsyntax-only %s -verify=expected,hlsl2021
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -std=hlsl202x -o - -fsyntax-only %s -verify=available
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -std=hlsl202y -o - -fsyntax-only %s -verify=available

// available-no-diagnostics

// hlsl2017-warning@*{{support for HLSL language version hlsl2017 is incomplete, recommend using hlsl202x instead}}
// hlsl2018-warning@*{{support for HLSL language version hlsl2018 is incomplete, recommend using hlsl202x instead}}

using namespace hlsl;

// expected-warning@+1{{'auto' type specifier is an HLSL 202x extension}}
auto foo(float A) -> float {
  return A;
}

void bar(float A) {
// expected-warning@+1{{'auto' type specifier is an HLSL 202x extension}}
  auto C = A;
}
