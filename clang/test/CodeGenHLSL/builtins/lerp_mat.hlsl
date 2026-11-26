// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -fnative-half-type -fnative-int16-type -emit-llvm -o - \
// RUN:   | FileCheck %s

// CHECK-LABEL: test_lerp_half2x3
// CHECK: [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <6 x half> %{{.*}}, %{{.*}}
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <6 x half> %{{.*}}, [[SUB]]
// CHECK: fadd reassoc nnan ninf nsz arcp afn <6 x half> %{{.*}}, [[MUL]]
half2x3 test_lerp_half2x3(half2x3 p0, half2x3 p1, half2x3 p2) { return lerp(p0, p1, p2); }

// CHECK-LABEL: test_lerp_half3x4
// CHECK: [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <12 x half> %{{.*}}, %{{.*}}
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <12 x half> %{{.*}}, [[SUB]]
// CHECK: fadd reassoc nnan ninf nsz arcp afn <12 x half> %{{.*}}, [[MUL]]
half3x4 test_lerp_half3x4(half3x4 p0, half3x4 p1, half3x4 p2) { return lerp(p0, p1, p2); }

// CHECK-LABEL: test_lerp_half4x4
// CHECK: [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <16 x half> %{{.*}}, %{{.*}}
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <16 x half> %{{.*}}, [[SUB]]
// CHECK: fadd reassoc nnan ninf nsz arcp afn <16 x half> %{{.*}}, [[MUL]]
half4x4 test_lerp_half4x4(half4x4 p0, half4x4 p1, half4x4 p2) { return lerp(p0, p1, p2); }

// CHECK-LABEL: test_lerp_float2x3
// CHECK: [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <6 x float> %{{.*}}, %{{.*}}
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <6 x float> %{{.*}}, [[SUB]]
// CHECK: fadd reassoc nnan ninf nsz arcp afn <6 x float> %{{.*}}, [[MUL]]
float2x3 test_lerp_float2x3(float2x3 p0, float2x3 p1, float2x3 p2) { return lerp(p0, p1, p2); }

// CHECK-LABEL: test_lerp_float3x4
// CHECK: [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <12 x float> %{{.*}}, %{{.*}}
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <12 x float> %{{.*}}, [[SUB]]
// CHECK: fadd reassoc nnan ninf nsz arcp afn <12 x float> %{{.*}}, [[MUL]]
float3x4 test_lerp_float3x4(float3x4 p0, float3x4 p1, float3x4 p2) { return lerp(p0, p1, p2); }

// CHECK-LABEL: test_lerp_float4x4
// CHECK: [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <16 x float> %{{.*}}, %{{.*}}
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <16 x float> %{{.*}}, [[SUB]]
// CHECK: fadd reassoc nnan ninf nsz arcp afn <16 x float> %{{.*}}, [[MUL]]
float4x4 test_lerp_float4x4(float4x4 p0, float4x4 p1, float4x4 p2) { return lerp(p0, p1, p2); }
