// RUN: %clang_cc1 -triple spir-unknown-unknown -cl-std=CL3.0 -fdeclare-opencl-builtins -verify -fsyntax-only %s

// Keep this test header-free so it exercises OpenCLBuiltins.td instead of
// declarations from opencl-c.h.

typedef unsigned short ushort;
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float16 __attribute__((ext_vector_type(16)));
typedef ushort ushort2 __attribute__((ext_vector_type(2)));
typedef ushort ushort3 __attribute__((ext_vector_type(3)));
typedef ushort ushort4 __attribute__((ext_vector_type(4)));
typedef ushort ushort8 __attribute__((ext_vector_type(8)));
typedef ushort ushort16 __attribute__((ext_vector_type(16)));

ushort test_convert_bfloat16_as_ushort(float source) {
  return intel_convert_bfloat16_as_ushort(source);
}

ushort2 test_convert_bfloat162_as_ushort2(float2 source) {
  return intel_convert_bfloat162_as_ushort2(source);
}

ushort3 test_convert_bfloat163_as_ushort3(float3 source) {
  return intel_convert_bfloat163_as_ushort3(source);
}

ushort4 test_convert_bfloat164_as_ushort4(float4 source) {
  return intel_convert_bfloat164_as_ushort4(source);
}

ushort8 test_convert_bfloat168_as_ushort8(float8 source) {
  return intel_convert_bfloat168_as_ushort8(source);
}

ushort16 test_convert_bfloat1616_as_ushort16(float16 source) {
  return intel_convert_bfloat1616_as_ushort16(source);
}

float test_convert_as_bfloat16_float(ushort source) {
  return intel_convert_as_bfloat16_float(source);
}

float2 test_convert_as_bfloat162_float2(ushort2 source) {
  return intel_convert_as_bfloat162_float2(source);
}

float3 test_convert_as_bfloat163_float3(ushort3 source) {
  return intel_convert_as_bfloat163_float3(source);
}

float4 test_convert_as_bfloat164_float4(ushort4 source) {
  return intel_convert_as_bfloat164_float4(source);
}

float8 test_convert_as_bfloat168_float8(ushort8 source) {
  return intel_convert_as_bfloat168_float8(source);
}

float16 test_convert_as_bfloat1616_float16(ushort16 source) {
  return intel_convert_as_bfloat1616_float16(source);
}

struct S { int x; };

void test_convert_bfloat16_as_ushort_invalid(float source, struct S s,
                                             float4 f4) {
  intel_convert_bfloat16_as_ushort(); // expected-error{{too few arguments to function call, expected 1, have 0}}
  // expected-note@-1 0+{{'intel_convert_bfloat16_as_ushort' declared here}}
  intel_convert_bfloat16_as_ushort(source, source); // expected-error{{too many arguments to function call, expected 1, have 2}}
  // expected-note@-1 0+{{'intel_convert_bfloat16_as_ushort' declared here}}
  intel_convert_bfloat16_as_ushort(s); // expected-error{{passing '__private struct S' to parameter of incompatible type 'float'}}
  intel_convert_bfloat162_as_ushort2(f4); // expected-error{{passing '__private float4' (vector of 4 'float' values) to parameter of incompatible type 'float __attribute__((ext_vector_type(2)))' (vector of 2 'float' values)}}
}

void test_convert_as_bfloat16_float_invalid(ushort source, struct S s,
                                            ushort4 u4) {
  intel_convert_as_bfloat16_float(); // expected-error{{too few arguments to function call, expected 1, have 0}}
  // expected-note@-1 0+{{'intel_convert_as_bfloat16_float' declared here}}
  intel_convert_as_bfloat16_float(source, source); // expected-error{{too many arguments to function call, expected 1, have 2}}
  // expected-note@-1 0+{{'intel_convert_as_bfloat16_float' declared here}}
  intel_convert_as_bfloat16_float(s); // expected-error{{passing '__private struct S' to parameter of incompatible type 'unsigned short'}}
  intel_convert_as_bfloat162_float2(u4); // expected-error{{passing '__private ushort4' (vector of 4 'ushort' values) to parameter of incompatible type 'unsigned short __attribute__((ext_vector_type(2)))' (vector of 2 'unsigned short' values)}}
}
