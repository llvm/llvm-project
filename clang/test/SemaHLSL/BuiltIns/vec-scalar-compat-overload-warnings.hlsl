// RUN: %clang_cc1 -finclude-default-header -triple dxilv1.0-unknown-shadermodel6.0-compute -std=hlsl202x -emit-llvm-only -disable-llvm-passes %s 2>&1 | FileCheck %s

float2 clamp_test1(float2 p0, float2 p1, float p2) {
  // CHECK: warning: 'clamp<float, 2U>' is deprecated: In 202x mismatched vector/scalar lowering for clamp is deprecated. Explicitly cast parameters.
  return clamp(p0, p1, p2);
}

float3 clamp_test2(float3 p0, float p1, float3 p2) {
  // CHECK: warning: 'clamp<float, 3U>' is deprecated: In 202x mismatched vector/scalar lowering for clamp is deprecated. Explicitly cast parameters.
  return clamp(p0, p1, p2);
}

float4 clamp_test3(float4 p0, float p1, float p2) {
  // CHECK: warning: 'clamp<float, 4U>' is deprecated: In 202x mismatched vector/scalar lowering for clamp is deprecated. Explicitly cast parameters.
  return clamp(p0, p1, p2);
}

float3 lerp_test(float3 p0, float3 p1, float p2) {
  // CHECK: warning: 'lerp<float, 3U>' is deprecated: In 202x mismatched vector/scalar lowering for lerp is deprecated. Explicitly cast parameters.
  return lerp(p0, p1, p2);
}

float2 max_test1(float2 p0, float p1) {
  // CHECK: warning: 'max<float, 2U>' is deprecated: In 202x mismatched vector/scalar lowering for max is deprecated. Explicitly cast parameters.
  return max(p0, p1);
}

float3 max_test2(float p0, float3 p1) {
  // CHECK: warning: 'max<float, 3U>' is deprecated: In 202x mismatched vector/scalar lowering for max is deprecated. Explicitly cast parameters.
  return max(p0, p1);
}

float2 min_test1(float2 p0, float p1) {
  // CHECK: warning: 'min<float, 2U>' is deprecated: In 202x mismatched vector/scalar lowering for min is deprecated. Explicitly cast parameters.
  return min(p0, p1);
}

float3 min_test2(float p0, float3 p1) {
  // CHECK: warning: 'min<float, 3U>' is deprecated: In 202x mismatched vector/scalar lowering for min is deprecated. Explicitly cast parameters.
  return min(p0, p1);
}
