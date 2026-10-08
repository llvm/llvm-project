// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -O1 -o - | FileCheck %s

// CHECK-LABEL: test_step_half
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt half %p1, %p0
// CHECK-NEXT: [[SELECT:%.*]] = select reassoc nnan ninf nsz arcp afn i1 [[CMP]], half 0.000000e+00, half 1.000000e+00
// CHECK-NEXT: ret half [[SELECT]]
half test_step_half(half p0, half p1)
{
    return step(p0, p1);
}
// CHECK-LABEL: test_step_half2
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <2 x half> %p1, %p0
// CHECK-NEXT: [[SELECT:%.*]] = select reassoc nnan ninf nsz arcp afn <2 x i1> [[CMP]], <2 x half> zeroinitializer, <2 x half> splat (half 1.000000e+00)
// CHECK-NEXT: ret <2 x half> [[SELECT]]
half2 test_step_half2(half2 p0, half2 p1)
{
    return step(p0, p1);
}
// CHECK-LABEL: test_step_half3
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <3 x half> %p1, %p0
// CHECK-NEXT: [[SELECT:%.*]] = select reassoc nnan ninf nsz arcp afn <3 x i1> [[CMP]], <3 x half> zeroinitializer, <3 x half> splat (half 1.000000e+00)
// CHECK-NEXT: ret <3 x half> [[SELECT]]
half3 test_step_half3(half3 p0, half3 p1)
{
    return step(p0, p1);
}
// CHECK-LABEL: test_step_half4
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <4 x half> %p1, %p0
// CHECK-NEXT: [[SELECT:%.*]] = select reassoc nnan ninf nsz arcp afn <4 x i1> [[CMP]], <4 x half> zeroinitializer, <4 x half> splat (half 1.000000e+00)
// CHECK-NEXT: ret <4 x half> [[SELECT]]
half4 test_step_half4(half4 p0, half4 p1)
{
    return step(p0, p1);
}

// CHECK-LABEL: test_step_float
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt float %p1, %p0
// CHECK-NEXT: [[SELECT:%.*]] = select reassoc nnan ninf nsz arcp afn i1 [[CMP]], float 0.000000e+00, float 1.000000e+00
// CHECK-NEXT: ret float [[SELECT]]
float test_step_float(float p0, float p1)
{
    return step(p0, p1);
}
// CHECK-LABEL: test_step_float2
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <2 x float> %p1, %p0
// CHECK-NEXT: [[SELECT:%.*]] = select reassoc nnan ninf nsz arcp afn <2 x i1> [[CMP]], <2 x float> zeroinitializer, <2 x float> splat (float 1.000000e+00)
// CHECK-NEXT: ret <2 x float> [[SELECT]]
float2 test_step_float2(float2 p0, float2 p1)
{
    return step(p0, p1);
}
// CHECK-LABEL: test_step_float3
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <3 x float> %p1, %p0
// CHECK-NEXT: [[SELECT:%.*]] = select reassoc nnan ninf nsz arcp afn <3 x i1> [[CMP]], <3 x float> zeroinitializer, <3 x float> splat (float 1.000000e+00)
// CHECK-NEXT: ret <3 x float> [[SELECT]]
float3 test_step_float3(float3 p0, float3 p1)
{
    return step(p0, p1);
}
// CHECK-LABEL: test_step_float4
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <4 x float> %p1, %p0
// CHECK-NEXT: [[SELECT:%.*]] = select reassoc nnan ninf nsz arcp afn <4 x i1> [[CMP]], <4 x float> zeroinitializer, <4 x float> splat (float 1.000000e+00)
// CHECK-NEXT: ret <4 x float> [[SELECT]]
float4 test_step_float4(float4 p0, float4 p1)
{
    return step(p0, p1);
}

// CHECK-LABEL: test_step_float5
// CHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <5 x float> %p1, %p0
// CHECK-NEXT: [[SELECT:%.*]] = select reassoc nnan ninf nsz arcp afn <5 x i1> [[CMP]], <5 x float> zeroinitializer, <5 x float> splat (float 1.000000e+00)
// CHECK-NEXT: ret <5 x float> [[SELECT]]
vector<float, 5> test_step_float5(vector<float, 5> p0,
                                  vector<float, 5> p1) {
    return step(p0, p1);
}
