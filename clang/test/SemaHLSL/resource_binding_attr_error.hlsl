// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// expected-error@+5 {{expected ';' after top level declarator}}
// expected-error@+4 {{expected ')'}}
// expected-note@+3 {{to match this '('}}
// expected-error@+2 {{a type specifier is required for all declarations}}
// expected-error@+1 {{illegal storage class on file-scoped variable}}
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
