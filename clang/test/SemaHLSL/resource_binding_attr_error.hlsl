// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

template<typename T>
struct MyTemplatedSRV {
  [[hlsl::resource_class(SRV)]] T x;
};

// valid, The register keyword in this statement isn't binding a resource, rather it is
// specifying a constant register binding offset within the $Globals cbuffer, which is legacy behavior from DX9.
float a : register(c0);

// expected-error@+1 {{binding type 'i' ignored. The 'integer constant' binding type is no longer supported}}
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
static MyTemplatedSRV<float> U : register(u5);

// expected-error@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
static float sa : register(c1);

float x[2] : register(c2); // valid
float y[2][2] : register(c3); // valid
float z[2][2][3] : register(c4); // valid

// expected-error@+1 {{binding type 'c' only applies to numeric variables in the global scope}}
groupshared float fa[10] : register(c5);

void foo() {
  // expected-error@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
  MyTemplatedSRV<float> U : register(u3);
}
void foo2() {
  // expected-error@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
  extern MyTemplatedSRV<float> U2 : register(u5);
}

// expected-error@+1 {{binding type 'u' only applies to UAV resources}}
float b : register(u0, space1);

// expected-error@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
void bar(MyTemplatedSRV<float> U : register(u3)) {

}

struct S {  
  // expected-error@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
  MyTemplatedSRV<float> U : register(u3);
};

// expected-error@+1 {{binding type 'z' is invalid}}
MyTemplatedSRV<float> U3 : register(z5);
