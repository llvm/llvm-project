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