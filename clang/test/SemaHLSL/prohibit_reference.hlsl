// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -o - -fsyntax-only %s -verify

int& bark(int); // expected-error {{references are unsupported in HLSL}}
void meow(int&); // expected-error {{references are unsupported in HLSL}}
void chirp(int &&); // expected-error {{references are unsupported in HLSL}}
// expected-warning@-1 {{rvalue references are a C++11 extension}}

struct Foo {
  int X;
  int Y;
};

int entry() {
  int X;
  int &Y = X; // expected-error {{references are unsupported in HLSL}}
}

int roar(Foo &F) { // expected-error {{references are unsupported in HLSL}}
  return F.X;
}
