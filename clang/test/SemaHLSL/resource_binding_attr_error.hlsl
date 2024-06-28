// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// FIXME: emit a diagnostic because float doesn't match the 'c' register type
float a : register(c0, space1);

// FIXME: emit a diagnostic because cbuffer doesn't match the 'i' register type
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

// expected-error@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
static RWBuffer<float> U : register(u5);

void foo() {
  // expected-error@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
  RWBuffer<float> U : register(u3);
}
void foo2() {
  // expected-error@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
  extern RWBuffer<float> U2 : register(u5);
}

// FIXME: emit a diagnostic because float doesn't match the 'u' register type
float b : register(u0, space1);

// expected-error@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
void bar(RWBuffer<float> U : register(u3)) {

}

struct S {  
  // expected-error@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
  RWBuffer<float> U : register(u3);
};
