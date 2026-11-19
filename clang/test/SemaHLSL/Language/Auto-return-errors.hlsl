// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -std=hlsl202x -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -std=hlsl202y -fsyntax-only -verify %s

auto forward(); // expected-note {{declared here}}

void use_before_definition() {
  forward(); // expected-error {{function 'forward' with deduced return type cannot be used before it is defined}}
}

auto forward() {
  return 42;
}
