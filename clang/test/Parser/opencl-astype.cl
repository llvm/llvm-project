// RUN: %clang_cc1 -fsyntax-only -verify %s -triple spir-unknown-unknown
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

void test_astype() {
  float f = 1.0f;
  unsigned int i = __builtin_astype(f, unsigned int); 
  
  typedef __attribute__(( ext_vector_type(4) ))  int int4;
  typedef __attribute__(( ext_vector_type(3) ))  float float3;
  typedef __attribute__(( ext_vector_type(4) ))  float float4;
  typedef __attribute__(( ext_vector_type(4) ))  double double4;
  
  float4 f4;
  double4 d4 = __builtin_astype(f4, double4); // expected-error{{invalid reinterpretation: sizes of 'double4' (vector of 4 'double' values) and '__private float4' (vector of 4 'float' values) must match}}
  
  // Verify int4->float3, float3->int4 works.
  int4 i4;
  float3 f3 = __builtin_astype(i4, float3);
  i4 = __builtin_astype(f3, int4);
}
