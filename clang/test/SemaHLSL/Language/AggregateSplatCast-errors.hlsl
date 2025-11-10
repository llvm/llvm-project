// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify -verify-ignore-unexpected=note

struct S {
  int A : 8;
  int B;
};

struct R {
  int A;
  union {
    float F;
    int4 G;
  };
};

// Can't cast a union
export void cantCast2() {
  R r = (R)1;
  // expected-error@-1 {{no matching conversion for C-style cast from 'int' to 'R'}}
}

RWBuffer<float4> Buf;

// Can't cast an intangible type
export void cantCast3() {
  Buf = (RWBuffer<float4>)1;
  // expected-error@-1 {{no matching conversion for C-style cast from 'int' to 'RWBuffer<float4>' (aka 'RWBuffer<vector<float, 4>>')}}
}

export void cantCast4() {
 RWBuffer<float4> B[2] = (RWBuffer<float4>[2])1;
 // expected-error@-1 {{C-style cast from 'int' to 'RWBuffer<float4>[2]' (aka 'RWBuffer<vector<float, 4>>[2]') is not allowed}}
}

struct X {
  int A;
  RWBuffer<float4> Buf;
};

export void cantCast5() {
  X x = (X)1;
  // expected-error@-1 {{no matching conversion for C-style cast from 'int' to 'X'}}
}
