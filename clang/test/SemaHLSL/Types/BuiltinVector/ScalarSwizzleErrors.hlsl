// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -verify %s

int2 ToTwoInts(int V) {
  return V.xy; // expected-error{{vector component access exceeds type 'vector<int, 1>' (vector of 1 'int' value)}}
}

float2 ToTwoFloats(float V) {
  return V.rg; // expected-error{{vector component access exceeds type 'vector<float, 1>' (vector of 1 'float' value)}}
}

int4 SomeNonsense(int V) {
  return V.poop; // expected-error{{illegal vector component name 'p'}}
}

float2 WhatIsHappening(float V) {
  return V.; // expected-error{{expected unqualified-id}}
}

float ScalarLValue(float2 V) {
  (float)V = 4.0; // expected-error{{assignment to cast is illegal, lvalue casts are not supported}}
}

// These cases produce no error.

float2 HowManyFloats(float V) {
  return V.rr.rr;
}

int64_t4 HooBoy() {
  return 4l.xxxx;
}

float3 AllRighty() {
  return 1..rrr;
}
