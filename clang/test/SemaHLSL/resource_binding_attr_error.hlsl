// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// expected-error@+1 {{invalid resource class specifier 'c' used; expected 'b', 's', 't', or 'u'}}
float a : register(c0, space1);

// expected-error@+1 {{invalid resource class specifier 'i' used; expected 'b', 's', 't', or 'u'}}
cbuffer b : register(i0) {

}
// expected-error@+1 {{invalid space specifier 's2' used; expected 'space' followed by an integer, like space1}}
cbuffer c : register(b0, s2) {

}
// expected-error@+1 {{register number should be an integer}}
cbuffer d : register(bf, s2) {

}
// expected-error@+1 {{invalid space specifier 'spaces' used; expected 'space' followed by an integer, like space1}}
cbuffer e : register(b2, spaces) {

}

// expected-error@+1 {{expected identifier}}
cbuffer A : register() {}

// expected-error@+1 {{register number should be an integer}}
cbuffer B : register(space1) {}

// expected-error@+1 {{wrong argument format for hlsl attribute, use b2 instead}}
cbuffer C : register(b 2) {}

// expected-error@+2 {{wrong argument format for hlsl attribute, use b2 instead}}
// expected-error@+1 {{wrong argument format for hlsl attribute, use space3 instead}}
cbuffer D : register(b 2, space 3) {}

// expected-warning@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
static RWBuffer<float> U : register(u5);

void foo() {
  // expected-warning@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
  RWBuffer<float> U : register(u3);
}
void foo2() {
  // expected-warning@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
  extern RWBuffer<float> U2 : register(u5);
}
// FIXME: expect-error once fix https://github.com/llvm/llvm-project/issues/57886.
float b : register(u0, space1);

// expected-warning@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
void bar(RWBuffer<float> U : register(u3)) {

}

struct S {
  // FIXME: generate better error when support semantic on struct field.
  // See https://github.com/llvm/llvm-project/issues/57889.
  // expected-error@+1 {{expected expression}}
  RWBuffer<float> U : register(u3);
};
