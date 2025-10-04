// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify %s

typedef float float2x1 __attribute__((matrix_type(2,1)));
typedef float float2x2 __attribute__((matrix_type(2,2)));
typedef float float2 __attribute__((ext_vector_type(2)));

struct S { float f; };
struct S2 { float2 f;};

[numthreads(1,1,1)]
void entry() {
 float2x1 LilMat = float2x1(1.0, 2.0);
 float2x1 BrokenMat = float2x1(1.0, 2.0, 3.0); // expected-error{{excess elements in matrix initializer}}
 float2x2 NormieMat = float2x2(LilMat, 3.0, 4.0, 5.0); // expected-error{{excess elements in matrix initializer}}
 float2x2 BrokenNormie = float2x2(3.0, 4.0); // expected-error{{too few elements in matrix initialization (expected 4 elements, have 2)}}
 float2x1 OverwhemledNormie = float2x1(3.0, 4.0, 5.0, 6.0); // expected-error{{excess elements in matrix initializer}}

  // These _should_ work in HLSL but aren't yet supported.
  S s;
  float2x1 GettingStrange = float2x1(s, s); // expected-error{{no viable conversion from 'S' to 'float'}} expected-error{{no viable conversion from 'S' to 'float'}}

  S2 s2;
  float2x2 GettingStrange2 = float2x2(s2, s2); // expected-error{{no viable conversion from 'S2' to 'float'}} expected-error{{no viable conversion from 'S2' to 'float'}} expected-error{{too few elements in matrix initialization (expected 4 elements, have 2)}}
}
