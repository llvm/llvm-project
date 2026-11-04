// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -fnative-half-type -fnative-int16-type -emit-llvm -o - \
// RUN:   | FileCheck %s

// CHECK-LABEL: test_step_half2x3
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <6 x half> %{{.*}}, %{{.*}}
// CHECK: select reassoc nnan ninf nsz arcp afn <6 x i1> [[CMP]], <6 x half> zeroinitializer, <6 x half> splat (half 1.000000e+00)
half2x3 test_step_half2x3(half2x3 p0, half2x3 p1) { return step(p0, p1); }

// CHECK-LABEL: test_step_half3x4
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <12 x half> %{{.*}}, %{{.*}}
// CHECK: select reassoc nnan ninf nsz arcp afn <12 x i1> [[CMP]], <12 x half> zeroinitializer, <12 x half> splat (half 1.000000e+00)
half3x4 test_step_half3x4(half3x4 p0, half3x4 p1) { return step(p0, p1); }

// CHECK-LABEL: test_step_half4x4
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <16 x half> %{{.*}}, %{{.*}}
// CHECK: select reassoc nnan ninf nsz arcp afn <16 x i1> [[CMP]], <16 x half> zeroinitializer, <16 x half> splat (half 1.000000e+00)
half4x4 test_step_half4x4(half4x4 p0, half4x4 p1) { return step(p0, p1); }

// CHECK-LABEL: test_step_float2x3
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <6 x float> %{{.*}}, %{{.*}}
// CHECK: select reassoc nnan ninf nsz arcp afn <6 x i1> [[CMP]], <6 x float> zeroinitializer, <6 x float> splat (float 1.000000e+00)
float2x3 test_step_float2x3(float2x3 p0, float2x3 p1) { return step(p0, p1); }

// CHECK-LABEL: test_step_float3x4
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <12 x float> %{{.*}}, %{{.*}}
// CHECK: select reassoc nnan ninf nsz arcp afn <12 x i1> [[CMP]], <12 x float> zeroinitializer, <12 x float> splat (float 1.000000e+00)
float3x4 test_step_float3x4(float3x4 p0, float3x4 p1) { return step(p0, p1); }

// CHECK-LABEL: test_step_float4x4
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <16 x float> %{{.*}}, %{{.*}}
// CHECK: select reassoc nnan ninf nsz arcp afn <16 x i1> [[CMP]], <16 x float> zeroinitializer, <16 x float> splat (float 1.000000e+00)
float4x4 test_step_float4x4(float4x4 p0, float4x4 p1) { return step(p0, p1); }
