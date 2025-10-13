// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only -Wno-legacy-constant-register-binding %s -verify

// expected-no-diagnostics
float f2 : register(b9);

float f3 : register(i9);

cbuffer g_cbuffer1 {
    float f4 : register(c2);
};


struct Eg12{
  RWBuffer<int> a;
};

Eg12 e12 : register(c9);

Eg12 bar : register(i1);

struct Eg7 {
  struct Bar {
    float f;
  };
  Bar b;
};
Eg7 e7 : register(t0);
