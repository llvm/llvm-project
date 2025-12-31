// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// test semantic validation for register numbers that exceed UINT32_MAX

struct S {
  RWBuffer<float> A[4];
  RWBuffer<int> B[10];
};

// do some more nesting
struct S2 {
  S a[3];
};

// test that S.A carries the register number over the limit and emits the error
// expected-error@+1 {{register number should not exceed 4294967295}}
S s : register(u4294967294); // UINT32_MAX - 1

// test the error is also triggered when analyzing S.B
// expected-error@+1 {{register number should not exceed 4294967295}}
S s2 : register(u4294967289);


// test the error is also triggered when analyzing S2.a[1].B
// expected-error@+1 {{register number should not exceed 4294967295}}
S2 s3 : register(u4294967275);

// expected-error@+1 {{register number should not exceed 4294967295}}
RWBuffer<float> Buf[10][10] : register(u4294967234);

// test a standard resource array
// expected-error@+1 {{register number should not exceed 4294967295}}
RWBuffer<float> Buf2[10] : register(u4294967294); 

// test directly an excessively high register number.
// expected-error@+1 {{register number should not exceed 4294967295}}
RWBuffer<float> A : register(u9995294967294);

// test a cbuffer directly with an excessively high register number.
// expected-error@+1 {{register number should not exceed 4294967295}}
cbuffer MyCB : register(b9995294967294) {
  float F[4];
  int   I[10];
};

// no errors expected, all 100 register numbers are occupied here
RWBuffer<float> Buf3[10][10] : register(u4294967194); 

// expected-error@+1 {{register number should be an integer}}
RWBuffer<float> Buf4[10][10] : register(ud); 
