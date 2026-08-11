// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fsyntax-only -verify -verify-ignore-unexpected=warning

cbuffer MatArr0Pass {
  float2x4 A0p[2] : packoffset(c0.x);
  float    a0tail : packoffset(c4.x);
}

cbuffer MatArr0Fail {
  float2x4 A0f[2] : packoffset(c0.x);
  float    a0bad  : packoffset(c3.z);
  // expected-error@-1 {{packoffset overlap between 'a0bad', 'A0f'}}
}

// Struct containing a matrix.

struct MS0 {
  float2x4 M;
  float2   V;
};

cbuffer MatStruct0Pass {
  MS0   s0p   : packoffset(c0.x);
  float s0tail: packoffset(c2.z);
}

cbuffer MatStruct0Fail {
  MS0   s0f   : packoffset(c0.x);
  float s0bad : packoffset(c2.y);
  // expected-error@-1 {{packoffset overlap between 's0bad', 's0f'}}
}

// Nested struct containing a matrix.
struct Inner0 {
  float2x4 M;
  float    F;
};

struct Outer0 {
  float2   Head;
  Inner0   I;
  float2   Tail;
};

cbuffer MatNested0Pass {
  Outer0 o0p   : packoffset(c0.x);
  float  o0tail: packoffset(c4.x);
}

cbuffer MatNested0Fail {
  Outer0 o0f  : packoffset(c0.x);
  float  o0bad: packoffset(c3.z);
  // expected-error@-1 {{packoffset overlap between 'o0bad', 'o0f'}}
}

// Array-of-struct where struct contains a matrix.

struct AMS0 {
  float2x4 M;
  float2   V;
};

cbuffer MatArrStruct0Pass {
  AMS0  as0p[2] : packoffset(c0.x);
  float as0tail : packoffset(c5.z);
}

cbuffer MatArrStruct0Fail {
  AMS0  as0f[2] : packoffset(c0.x);
  float as0bad  : packoffset(c5.y);
  // expected-error@-1 {{packoffset overlap between 'as0bad', 'as0f'}}
}
