// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -verify %s

struct FirstBase {
  int X;
};

struct SecondBase {
  int Y;
};

struct SingleInheritance : FirstBase {};
struct ChainedInheritance : SingleInheritance {};

struct MultipleInheritance : FirstBase, SecondBase {}; // expected-error{{HLSL does not support multiple inheritance}}

struct MultiError : FirstBase, FirstBase {}; // expected-error{{HLSL does not support multiple inheritance}}
