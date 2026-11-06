// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -fnative-half-type -fnative-int16-type -emit-llvm -o - \
// RUN:   | FileCheck %s

// CHECK-LABEL: test_degrees_half1x2
// CHECK: fmul reassoc nnan ninf nsz arcp afn <2 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <2 x half>
half1x2 test_degrees_half1x2(half1x2 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half1x3
// CHECK: fmul reassoc nnan ninf nsz arcp afn <3 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <3 x half>
half1x3 test_degrees_half1x3(half1x3 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half1x4
// CHECK: fmul reassoc nnan ninf nsz arcp afn <4 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <4 x half>
half1x4 test_degrees_half1x4(half1x4 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half2x1
// CHECK: fmul reassoc nnan ninf nsz arcp afn <2 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <2 x half>
half2x1 test_degrees_half2x1(half2x1 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half2x2
// CHECK: fmul reassoc nnan ninf nsz arcp afn <4 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <4 x half>
half2x2 test_degrees_half2x2(half2x2 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half2x3
// CHECK: fmul reassoc nnan ninf nsz arcp afn <6 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <6 x half>
half2x3 test_degrees_half2x3(half2x3 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half2x4
// CHECK: fmul reassoc nnan ninf nsz arcp afn <8 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <8 x half>
half2x4 test_degrees_half2x4(half2x4 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half3x1
// CHECK: fmul reassoc nnan ninf nsz arcp afn <3 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <3 x half>
half3x1 test_degrees_half3x1(half3x1 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half3x2
// CHECK: fmul reassoc nnan ninf nsz arcp afn <6 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <6 x half>
half3x2 test_degrees_half3x2(half3x2 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half3x3
// CHECK: fmul reassoc nnan ninf nsz arcp afn <9 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <9 x half>
half3x3 test_degrees_half3x3(half3x3 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half3x4
// CHECK: fmul reassoc nnan ninf nsz arcp afn <12 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <12 x half>
half3x4 test_degrees_half3x4(half3x4 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half4x1
// CHECK: fmul reassoc nnan ninf nsz arcp afn <4 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <4 x half>
half4x1 test_degrees_half4x1(half4x1 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half4x2
// CHECK: fmul reassoc nnan ninf nsz arcp afn <8 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <8 x half>
half4x2 test_degrees_half4x2(half4x2 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half4x3
// CHECK: fmul reassoc nnan ninf nsz arcp afn <12 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <12 x half>
half4x3 test_degrees_half4x3(half4x3 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_half4x4
// CHECK: fmul reassoc nnan ninf nsz arcp afn <16 x half> {{.*}}, splat (half 5.728130e+01)
// CHECK: ret <16 x half>
half4x4 test_degrees_half4x4(half4x4 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float1x2
// CHECK: fmul reassoc nnan ninf nsz arcp afn <2 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <2 x float>
float1x2 test_degrees_float1x2(float1x2 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float1x3
// CHECK: fmul reassoc nnan ninf nsz arcp afn <3 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <3 x float>
float1x3 test_degrees_float1x3(float1x3 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float1x4
// CHECK: fmul reassoc nnan ninf nsz arcp afn <4 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <4 x float>
float1x4 test_degrees_float1x4(float1x4 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float2x1
// CHECK: fmul reassoc nnan ninf nsz arcp afn <2 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <2 x float>
float2x1 test_degrees_float2x1(float2x1 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float2x2
// CHECK: fmul reassoc nnan ninf nsz arcp afn <4 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <4 x float>
float2x2 test_degrees_float2x2(float2x2 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float2x3
// CHECK: fmul reassoc nnan ninf nsz arcp afn <6 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <6 x float>
float2x3 test_degrees_float2x3(float2x3 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float2x4
// CHECK: fmul reassoc nnan ninf nsz arcp afn <8 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <8 x float>
float2x4 test_degrees_float2x4(float2x4 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float3x1
// CHECK: fmul reassoc nnan ninf nsz arcp afn <3 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <3 x float>
float3x1 test_degrees_float3x1(float3x1 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float3x2
// CHECK: fmul reassoc nnan ninf nsz arcp afn <6 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <6 x float>
float3x2 test_degrees_float3x2(float3x2 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float3x3
// CHECK: fmul reassoc nnan ninf nsz arcp afn <9 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <9 x float>
float3x3 test_degrees_float3x3(float3x3 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float3x4
// CHECK: fmul reassoc nnan ninf nsz arcp afn <12 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <12 x float>
float3x4 test_degrees_float3x4(float3x4 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float4x1
// CHECK: fmul reassoc nnan ninf nsz arcp afn <4 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <4 x float>
float4x1 test_degrees_float4x1(float4x1 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float4x2
// CHECK: fmul reassoc nnan ninf nsz arcp afn <8 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <8 x float>
float4x2 test_degrees_float4x2(float4x2 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float4x3
// CHECK: fmul reassoc nnan ninf nsz arcp afn <12 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <12 x float>
float4x3 test_degrees_float4x3(float4x3 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float4x4
// CHECK: fmul reassoc nnan ninf nsz arcp afn <16 x float> {{.*}}, splat (float f0x42652EE1)
// CHECK: ret <16 x float>
float4x4 test_degrees_float4x4(float4x4 p0) { return degrees(p0); }

