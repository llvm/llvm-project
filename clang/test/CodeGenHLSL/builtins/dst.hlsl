// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.2-library %s -fnative-half-type -emit-llvm -O1 -o - | FileCheck %s


// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z12dstWithFloatDv4_fS_(
// CHECK-SAME: <4 x float> noundef nofpclass(nan inf) [[P:%.*]], <4 x float> noundef nofpclass(nan inf) [[Q:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] { 
// CHECK: [[VECEXT:%.*]] = extractelement <4 x float> [[P]], i64 1
// CHECK-NEXT: [[VECEXT1:%.*]] = extractelement <4 x float> [[Q]], i64 1
// CHECK-NEXT: [[MULRES:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[VECEXT1]], [[VECEXT]]
// CHECK-NEXT: [[VECINIT:%.*]] = insertelement <4 x float> <float 1.000000e+00, float poison, float poison, float poison>, float [[MULRES]], i64 1
// CHECK-NEXT: [[VECINIT3:%.*]] = shufflevector <4 x float> [[VECINIT]], <4 x float> [[P]], <4 x i32> <i32 0, i32 1, i32 6, i32 poison>
// CHECK-NEXT: [[VECINIT5:%.*]] = shufflevector <4 x float> [[VECINIT3]], <4 x float> [[Q]], <4 x i32> <i32 0, i32 1, i32 2, i32 7>
// CHECK-NEXT: ret <4 x float> [[VECINIT5]]

float4 dstWithFloat(float4 p1, float4 p2)
{
    return dst(p1, p2);
}

// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x half> @_Z11dstwithHalfDv4_DhS_(
// CHECK-SAME: <4 x half> noundef nofpclass(nan inf) [[P:%.*]], <4 x half> noundef nofpclass(nan inf) [[Q:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK: [[VECEXT:%.*]] = extractelement <4 x half> [[P]], i64 1
// CHECK-NEXT: [[VECEXT1:%.*]] = extractelement <4 x half> [[Q]], i64 1
// CHECK-NEXT: [[MULRES:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[VECEXT1]], [[VECEXT]]
// CHECK-NEXT: [[VECINIT:%.*]] = insertelement <4 x half> <half 0xH3C00, half poison, half poison, half poison>, half [[MULRES]], i64 1
// CHECK-NEXT: [[VECINIT3:%.*]] = shufflevector <4 x half> [[VECINIT]], <4 x half> [[P]], <4 x i32> <i32 0, i32 1, i32 6, i32 poison>
// CHECK-NEXT: [[VECINIT5:%.*]] = shufflevector <4 x half> [[VECINIT3]], <4 x half> [[Q]], <4 x i32> <i32 0, i32 1, i32 2, i32 7>
// CHECK-NEXT: ret <4 x half> [[VECINIT5]]
half4 dstwithHalf(half4 p1, half4 p2)
{
    return dst(p1, p2);
}

// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x double> @_Z13dstWithDoubleDv4_dS_(
// CHECK-SAME: <4 x double> noundef nofpclass(nan inf) [[P:%.*]], <4 x double> noundef nofpclass(nan inf) [[Q:%.*]]) local_unnamed_addr #[[ATTR0]] { 
// CHECK: [[VECEXT:%.*]] = extractelement <4 x double> [[P]], i64 1
// CHECK-NEXT: [[VECEXT1:%.*]] = extractelement <4 x double> [[Q]], i64 1
// CHECK-NEXT: [[MULRES:%.*]] = fmul reassoc nnan ninf nsz arcp afn double [[VECEXT1]], [[VECEXT]]
// CHECK-NEXT: [[VECINIT:%.*]] = insertelement <4 x double> <double 1.000000e+00, double poison, double poison, double poison>, double [[MULRES]], i64 1
// CHECK-NEXT: [[VECINIT3:%.*]] = shufflevector <4 x double> [[VECINIT]], <4 x double> [[P]], <4 x i32> <i32 0, i32 1, i32 6, i32 poison>
// CHECK-NEXT: [[VECINIT5:%.*]] = shufflevector <4 x double> [[VECINIT3]], <4 x double> [[Q]], <4 x i32> <i32 0, i32 1, i32 2, i32 7>
// CHECK-NEXT: ret <4 x double> [[VECINIT5]]
double4 dstWithDouble(double4 p1, double4 p2)
{
    return dst(p1, p2);
}

// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z9testfloatff(
// CHECK-SAME: float noundef nofpclass(nan inf) [[P:%.*]], float noundef nofpclass(nan inf) [[Q:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK: [[MULRES:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[Q]], [[P]]
// CHECK-NEXT: [[VECINIT:%.*]] = insertelement <4 x float> <float 1.000000e+00, float poison, float poison, float poison>, float [[MULRES]], i64 1
// CHECK-NEXT: [[VECINIT3:%.*]] = insertelement <4 x float> [[VECINIT]], float [[P]], i64 2
// CHECK-NEXT: [[VECINIT5:%.*]] = insertelement <4 x float> [[VECINIT3]], float [[Q]], i64 3
// CHECK-NEXT:  ret <4 x float> [[VECINIT5]]
float4 testfloat(float a, float b)
{
    return dst(a, b);
}

// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z10testfloat4fDv4_f(
// CHECK-SAME: float noundef nofpclass(nan inf) [[P:%.*]], <4 x float> noundef nofpclass(nan inf) [[Q:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK: [[VECEXT1:%.*]] = extractelement <4 x float> [[Q:%.*]], i64 1
// CHECK-NEXT: [[MULRES:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[VECEXT1]], [[P]]
// CHECK-NEXT: [[VECINIT:%.*]] = insertelement <4 x float> <float 1.000000e+00, float poison, float poison, float poison>, float [[MULRES]], i64 1
// CHECK-NEXT: [[VECINIT3:%.*]] = insertelement <4 x float> [[VECINIT]], float %a, i64 2
// CHECK-NEXT: [[VECINIT5:%.*]] = shufflevector <4 x float> [[VECINIT3]], <4 x float> [[Q]], <4 x i32> <i32 0, i32 1, i32 2, i32 7>
// CHECK-NEXT:  ret <4 x float> [[VECINIT5]]
float4 testfloat4(float a, float4 b)
{
    return dst(a, b);
}

// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x half> @_Z21testRetTypeShriinkingDv4_fS_(
// CHECK-SAME: <4 x float> noundef nofpclass(nan inf) [[P:%.*]], <4 x float> noundef nofpclass(nan inf) [[Q:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK: [[VECEXT:%.*]] = extractelement <4 x float> [[P]], i64 1
// CHECK-NEXT: [[VECEXT1:%.*]] = extractelement <4 x float> [[Q]], i64 1
// CHECK-NEXT: [[MULRES:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[VECEXT1]], [[VECEXT]]
// CHECK-NEXT: [[VECINIT:%.*]] = insertelement <4 x float> <float 1.000000e+00, float poison, float poison, float poison>, float [[MULRES]], i64 1
// CHECK-NEXT: [[VECINIT3:%.*]] = shufflevector <4 x float> [[VECINIT]], <4 x float> %a, <4 x i32> <i32 0, i32 1, i32 6, i32 poison>
// CHECK-NEXT: [[VECINIT5:%.*]] = shufflevector <4 x float> %vecinit3.i, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
// CHECK-NEXT: [[CONV:%.*]] = fptrunc reassoc nnan ninf nsz arcp afn <4 x float> [[VECINIT5]] to <4 x half>
// CHECK-NEXT:  ret <4 x half> [[CONV]]
half4 testRetTypeShriinking(float4 a, float4 b)
{
    return dst(a, b);
}
