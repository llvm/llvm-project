// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify %s

typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));

struct S { float f; };
struct S2 { float f; int i; };

[numthreads(1,1,1)]
void entry() {
  float2 LilVec = float2(1.0, 2.0);
  float2 BrokenVec = float2(1.0, 2.0, 3.0); // expected-error{{too many initializers in list for type 'float2' (vector of 2 'float' values) (expected 2 but found 3)}}
  float3 NormieVec = float3(LilVec, 3.0, 4.0); // expected-error{{too many initializers in list for type 'float3' (vector of 3 'float' values) (expected 3 but found 4)}}
  float3 BrokenNormie = float3(3.0, 4.0); // expected-error{{too few initializers in list for type 'float3' (vector of 3 'float' values) (expected 3 but found 2)}}
  float3 OverwhemledNormie = float3(3.0, 4.0, 5.0, 6.0); // expected-error{{too many initializers in list for type 'float3' (vector of 3 'float' values) (expected 3 but found 4)}}

  // These next two are a bit strange, but are consistent with HLSL today.
  S s;
  float2 GettingStrange = float2(s, s);
  S2 s2 = {1.0f, 2};
  float2 AlsoStrange = float2(s2);

  float2 TooManyStruts = float2(s2, s); // expected-error{{too many initializers in list for type 'float2' (vector of 2 'float' values) (expected 2 but found 3)}}

  // HLSL does not yet allow user-defined conversions.
  struct T {
    operator float() const { return 1.0f; }
  } t;
  // TODO: Should this work? Today HLSL doesn't resolve user-defined conversions here, but we maybe should...
  float2 foo5 = float2(t, t); // expected-error{{too few initializers in list for type 'float2' (vector of 2 'float' values) (expected 2 but found 0)}}
}
