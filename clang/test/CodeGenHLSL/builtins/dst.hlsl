// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.2-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s


// CHECK-LABEL: linkonce_odr noundef nofpclass(nan inf) <4 x float> @_ZN4hlsl8__detail8dst_implIfEEDv4_T_S3_S3_(
// CHECK-SAME: <4 x float> noundef nofpclass(nan inf) [[P:%.*]], <4 x float> noundef nofpclass(nan inf) [[Q:%.*]]) #[[ATTR0:[0-9]+]] { 
// CHECK: [[VECEXT:%.*]] = extractelement <4 x float> [[PADDR:%.*]], i32 1
// CHECK: [[VECEXT1:%.*]] = extractelement <4 x float> [[QADDR:%.*]], i32 1
// CHECK: [[MULRES:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[VECEXT]], [[VECEXT1]]
// CHECK: [[VECINIT:%.*]] = insertelement <4 x float> <float 1.000000e+00, float poison, float poison, float poison>, float [[MULRES]], i32 1
// CHECK: [[VECINIT2:%.*]] = extractelement <4 x float> [[PADDR2:%.*]], i32 2
// CHECK: [[VECINIT3:%.*]] = insertelement <4 x float> [[VECINIT]], float [[VECINIT2]], i32 2
// CHECK: [[VECINIT4:%.*]] = extractelement <4 x float> [[QADDR3:%.*]], i32 3
// CHECK: [[VECINIT5:%.*]] = insertelement <4 x float> [[VECINIT3]], float [[VECINIT4]], i32 3
// CHECK-NEXT: store <4 x float> [[VECINIT5]], ptr [[DEST:%.*]], align 16
// CHECK-NEXT: [[RES:%.*]] = load <4 x float>, ptr [[DEST]], align 16
// CHECK-NEXT: ret <4 x float> [[RES]]
float4 dstWithFloat(float4 p1, float4 p2)
{
    return dst(p1, p2);
}

// CHECK-LABEL: define linkonce_odr noundef nofpclass(nan inf) <4 x half> @_ZN4hlsl8__detail8dst_implIDhEEDv4_T_S3_S3_(
// CHECK-SAME: <4 x half> noundef nofpclass(nan inf) [[P:%.*]], <4 x half> noundef nofpclass(nan inf) [[Q:%.*]]) #[[ATTR0]] {
// CHECK: [[VECEXT:%.*]] = extractelement <4 x half> [[PADDR:%.*]], i32 1
// CHECK: [[VECEXT1:%.*]] = extractelement <4 x half> [[QADDR:%.*]], i32 1
// CHECK: [[MULRES:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[VECEXT]], [[VECEXT1]]
// CHECK: [[VECINIT:%.*]] = insertelement <4 x half> <half 0xH3C00, half poison, half poison, half poison>, half [[MULRES]], i32 1
// CHECK: [[VECINIT2:%.*]] = extractelement <4 x half> [[PADDR2:%.*]], i32 2
// CHECK: [[VECINIT3:%.*]] = insertelement <4 x half> [[VECINIT]], half [[VECINIT2]], i32 2
// CHECK: [[VECINIT4:%.*]] = extractelement <4 x half> [[QADDR3:%.*]], i32 3
// CHECK: [[VECINIT5:%.*]] = insertelement <4 x half> [[VECINIT3]], half [[VECINIT4]], i32 3
// CHECK-NEXT: store <4 x half> [[VECINIT5]], ptr [[DEST:%.*]], align 8
// CHECK-NEXT: [[RES:%.*]] = load <4 x half>, ptr [[DEST]], align 8
// CHECK-NEXT: ret <4 x half> [[RES]]
half4 dstwithHalf(half4 p1, half4 p2)
{
    return dst(p1, p2);
}

// CHECK-LABEL: define linkonce_odr noundef nofpclass(nan inf) <4 x double> @_ZN4hlsl8__detail8dst_implIdEEDv4_T_S3_S3_(
// CHECK-SAME: <4 x double> noundef nofpclass(nan inf) [[P:%.*]], <4 x double> noundef nofpclass(nan inf) [[Q:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK: [[VECEXT:%.*]] = extractelement <4 x double> [[PADDR:%.*]], i32 1
// CHECK: [[VECEXT1:%.*]] = extractelement <4 x double> [[QADDR:%.*]], i32 1
// CHECK: [[MULRES:%.*]] = fmul reassoc nnan ninf nsz arcp afn double [[VECEXT]], [[VECEXT1]]
// CHECK: [[VECINIT:%.*]] = insertelement <4 x double> <double 1.000000e+00, double poison, double poison, double poison>, double [[MULRES]], i32 1
// CHECK: [[VECINIT2:%.*]] = extractelement <4 x double> [[PADDR2:%.*]], i32 2
// CHECK: [[VECINIT3:%.*]] = insertelement <4 x double> [[VECINIT]], double [[VECINIT2]], i32 2
// CHECK: [[VECINIT4:%.*]] = extractelement <4 x double> [[QADDR3:%.*]], i32 3
// CHECK: [[VECINIT5:%.*]] = insertelement <4 x double> [[VECINIT3]], double [[VECINIT4]], i32 3
// CHECK-NEXT: store <4 x double> [[VECINIT5]], ptr [[DEST:%.*]], align 32
// CHECK-NEXT: [[RES:%.*]] = load <4 x double>, ptr [[DEST]], align 32
// CHECK-NEXT: ret <4 x double> [[RES]]
double4 dstWithDouble(double4 p1, double4 p2)
{
    return dst(p1, p2);
}

