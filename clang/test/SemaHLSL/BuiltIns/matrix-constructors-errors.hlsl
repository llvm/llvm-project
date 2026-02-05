// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify %s

typedef float float2x1 __attribute__((matrix_type(2,1)));
typedef float float2x2 __attribute__((matrix_type(2,2)));
typedef float float2 __attribute__((ext_vector_type(2)));

struct S { float f; };
struct S2 { float2 f;};

[numthreads(1,1,1)]
void entry() {
 float2x1 LilMat = float2x1(1.0, 2.0);
 float2x1 BrokenMat = float2x1(1.0, 2.0, 3.0); // expected-error{{too many initializers in list for type 'float2x1' (aka 'matrix<float, 2, 1>') (expected 2 but found 3)}}
 float2x2 NormieMat = float2x2(LilMat, 3.0, 4.0, 5.0); // expected-error{{too many initializers in list for type 'float2x2' (aka 'matrix<float, 2, 2>') (expected 4 but found 5)}}
 float2x2 BrokenNormie = float2x2(3.0, 4.0); // expected-error{{too few initializers in list for type 'float2x2' (aka 'matrix<float, 2, 2>') (expected 4 but found 2)}}
 float2x1 OverwhemledNormie = float2x1(3.0, 4.0, 5.0, 6.0); // expected-error{{too many initializers in list for type 'float2x1' (aka 'matrix<float, 2, 1>') (expected 2 but found 4)}}

  // These should work in HLSL and not error
  S s;
  float2x1 GettingStrange = float2x1(s, s); 

  S2 s2;
  float2x2 GettingStrange2 = float2x2(s2, s2); 

  // HLSL does not yet allow user-defined conversions.
  struct T {
    operator float() const { return 1.0f; }
  } t;
  // TODO: Should this work? Today HLSL doesn't resolve user-defined conversions here, but we maybe should...
  float2x1 foo5 = float2x1(t, t); // expected-error{{too few initializers in list for type 'float2x1' (aka 'matrix<float, 2, 1>') (expected 2 but found 0)}}
}
