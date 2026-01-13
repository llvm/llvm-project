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

struct MySRV {
  __hlsl_resource_t [[hlsl::resource_class(SRV)]] x;
};

struct MySampler {
  __hlsl_resource_t [[hlsl::resource_class(Sampler)]] x;
};

struct MyUAV {
  __hlsl_resource_t [[hlsl::resource_class(UAV)]] x;
};

// test that different resource classes don't contribute to the
// maximum limit
struct MyResources {
  MySRV TheSRV[10]; // t
  MySampler TheSampler[20]; // s
  MyUAV TheUAV[40]; // u
};

// no failures here, since only the SRV contributes to the count,
// and the count + 10 does not exceed uint32 max.
MyResources M : register(t4294967284);

// three failures, since each of the three resources exceed the limit
// expected-error@+3 {{register number should not exceed 4294967295}}
// expected-error@+2 {{register number should not exceed 4294967295}}
// expected-error@+1 {{register number should not exceed 4294967295}}
MyResources M2 : register(t4294967294) : register(s4294967294) : register(u4294967294);

// one failure here, just because the final UAV exceeds the limit.
// expected-error@+1 {{register number should not exceed 4294967295}}
MyResources M3 : register(t2) : register(s3) : register(u4294967280);


// expected-error@+1 {{register number should be an integer}}
RWBuffer<float> Buf3[10][10] : register(ud); 

// this should work
RWBuffer<float> GoodBuf : register(u4294967295);

// no errors expected, all 100 register numbers are occupied here
RWBuffer<float> GoodBufArray[10][10] : register(u4294967194); 


