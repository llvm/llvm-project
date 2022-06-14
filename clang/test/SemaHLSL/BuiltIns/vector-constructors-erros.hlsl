// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify %s

typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));

struct S { float f; };
struct S2 { float f; int i; };

[numthreads(1,1,1)]
void entry() {
  float2 LilVec = float2(1.0, 2.0);
  float2 BrokenVec = float2(1.0, 2.0, 3.0); // expected-error{{excess elements in vector initializer}}
  float3 NormieVec = float3(LilVec, 3.0, 4.0); // expected-error{{excess elements in vector initializer}}
  float3 BrokenNormie = float3(3.0, 4.0); // expected-error{{too few elements in vector initialization (expected 3 elements, have 2)}}
  float3 OverwhemledNormie = float3(3.0, 4.0, 5.0, 6.0); // expected-error{{excess elements in vector initializer}}

  // These _should_ work in HLSL but aren't yet supported.
  S s;
  float2 GettingStrange = float2(s, s); // expected-error{{no viable conversion from 'S' to 'float'}} expected-error{{no viable conversion from 'S' to 'float'}}
  S2 s2;
  float2 EvenStranger = float2(s2); // expected-error{{no viable conversion from 'S2' to 'float'}} expected-error{{too few elements in vector initialization (expected 2 elements, have 1)}}
}
