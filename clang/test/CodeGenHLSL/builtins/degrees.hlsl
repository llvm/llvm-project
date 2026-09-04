// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -O1 -o - | FileCheck %s

// Note: the 5.728130e+01 and f0x42652EE1 constants below equal 180/Pi.

// CHECK-LABEL: test_degrees_half
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, 5.728130e+01
// CHECK-NEXT: ret half [[MUL]]
half test_degrees_half(half p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_half2
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x half> %{{.*}}, splat (half 5.728130e+01)
// CHECK-NEXT: ret <2 x half> [[MUL]]
half2 test_degrees_half2(half2 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_half3
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x half> %{{.*}}, splat (half 5.728130e+01)
// CHECK-NEXT: ret <3 x half> [[MUL]]
half3 test_degrees_half3(half3 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_half4
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x half> %{{.*}}, splat (half 5.728130e+01)
// CHECK-NEXT: ret <4 x half> [[MUL]]
half4 test_degrees_half4(half4 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, f0x42652EE1
// CHECK-NEXT: ret float [[MUL]]
float test_degrees_float(float p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_float2
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK-NEXT: ret <2 x float> [[MUL]]
float2 test_degrees_float2(float2 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_float3
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK-NEXT: ret <3 x float> [[MUL]]
float3 test_degrees_float3(float3 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_float4
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK-NEXT: ret <4 x float> [[MUL]]
float4 test_degrees_float4(float4 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_float5
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <5 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK-NEXT: ret <5 x float> [[MUL]]
vector<float, 5> test_degrees_float5(vector<float, 5> p0) {
	return degrees(p0);
}
