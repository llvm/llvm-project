// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// test semantic validation for register numbers that exceed UINT32_MAX

struct S {
  RWBuffer<float> A[4];
  RWBuffer<int> B[10];
};

// test that S.A carries the register number over the limit and emits the error
// expected-error@+1 {{register number should not exceed UINT32_MAX, 4294967295}}
S s : register(u4294967294); // UINT32_MAX - 1

// test the error is also triggered when analyzing S.B
// expected-error@+1 {{register number should not exceed UINT32_MAX, 4294967295}}
S s2 : register(u4294967289);

// test a standard resource array
// expected-error@+1 {{register number should not exceed UINT32_MAX, 4294967295}}
RWBuffer<float> Buf[10] : register(u4294967294); 

// test directly an excessively high register number.
// expected-error@+1 {{register number should not exceed UINT32_MAX, 4294967295}}
RWBuffer<float> A : register(u9995294967294);

// test a struct within a cbuffer
// expected-error@+1 {{register number should not exceed UINT32_MAX, 4294967295}}
cbuffer MyCB : register(b9995294967294) {
  float F[4];
  int   I[10];
};
