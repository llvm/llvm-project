// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -O1 -o - | FileCheck %s

// Note: the 1.745610e-02 and f0x3C8EFA35 constants below equal Pi/180.

// CHECK-LABEL: test_radians_half
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, 1.745610e-02
// CHECK-NEXT: ret half [[MUL]]
half test_radians_half(half p0) { return radians(p0); }
// CHECK-LABEL: test_radians_half2
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x half> %{{.*}}, splat (half 1.745610e-02)
// CHECK-NEXT: ret <2 x half> [[MUL]]
half2 test_radians_half2(half2 p0) { return radians(p0); }
// CHECK-LABEL: test_radians_half3
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x half> %{{.*}}, splat (half 1.745610e-02)
// CHECK-NEXT: ret <3 x half> [[MUL]]
half3 test_radians_half3(half3 p0) { return radians(p0); }
// CHECK-LABEL: test_radians_half4
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x half> %{{.*}}, splat (half 1.745610e-02)
// CHECK-NEXT: ret <4 x half> [[MUL]]
half4 test_radians_half4(half4 p0) { return radians(p0); }

// CHECK-LABEL: test_radians_float
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, f0x3C8EFA35
// CHECK-NEXT: ret float [[MUL]]
float test_radians_float(float p0) { return radians(p0); }
// CHECK-LABEL: test_radians_float2
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, splat (float f0x3C8EFA35)
// CHECK-NEXT: ret <2 x float> [[MUL]]
float2 test_radians_float2(float2 p0) { return radians(p0); }
// CHECK-LABEL: test_radians_float3
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, splat (float f0x3C8EFA35)
// CHECK-NEXT: ret <3 x float> [[MUL]]
float3 test_radians_float3(float3 p0) { return radians(p0); }
// CHECK-LABEL: test_radians_float4
// CHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, splat (float f0x3C8EFA35)
// CHECK-NEXT: ret <4 x float> [[MUL]]
float4 test_radians_float4(float4 p0) { return radians(p0); }
