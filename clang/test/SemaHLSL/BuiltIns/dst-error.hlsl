// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify

float4 test_too_many_arg(float4 p0)
{
    dst(p0, p0, p0);
  // expected-error@-1 {{no matching function for call to 'dst'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 3 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 3 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 3 were provided}}
}

float4 test_no_second_arg(float4 p0)
{
    return dst(p0);
  // expected-error@-1 {{no matching function for call to 'dst'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 1 was provided}}
}

float4 test_no_args()
{
    return dst();
    // expected-error@-1 {{no matching function for call to 'dst'}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 0 were provided}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 0 were provided}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 0 were provided}}
}

float4 test_3_components(float3 p0, float3 p1)
{
    return dst(p0, p1);
    // expected-error@-1 {{no matching function for call to 'dst'}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: no known conversion from 'vector<[...], 3>' to 'vector<[...], 4>' for 1st argument}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: no known conversion from 'vector<float, 3>' to 'vector<half, 4>' for 1st argument}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: no known conversion from 'vector<float, 3>' to 'vector<double, 4>' for 1st argument}}
}

float4 test_with_ambiguous_inp(double4 p0, float4 p1)
{
    return dst(p0, p1);
    // expected-error@-1 {{call to 'dst' is ambiguous}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
}

float2 test_with_return_float2(float4 p0, float4 p1)
{
    return dst(p0, p1);
    // expected-warning@-1 {{implicit conversion truncates vector: 'vector<float, 4>' (vector of 4 'float' values) to 'vector<float, 2>' (vector of 2 'float' values)}}
}

float4 test_with_ambigious_float4_double_inp(float4 p0, double p1)
{
    return dst(p0, p1);
    // expected-error@-1 {{call to 'dst' is ambiguous}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
}

float4 test_with_ambigious_double_float4_inp(double p0, float4 p1)
{
    return dst(p0, p1);
    // expected-error@-1 {{call to 'dst' is ambiguous}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
}

float4 test_with_ambigious_double4_float_inp(double4 p0, float p1)
{
    return dst(p0, p1);
    // expected-error@-1 {{call to 'dst' is ambiguous}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
}
