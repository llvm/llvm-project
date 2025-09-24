// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.2-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s


// CHECK-LABEL: define {{.*}} <4 x float> @{{[A-Za-z1-9_]+}}dst_impl{{[A-Za-z1-9_]*}}(
// CHECK-SAME: <4 x float> {{[A-Za-z )(]*}} [[P:%.*]], <4 x float> {{[A-Za-z )(]*}} [[Q:%.*]]) #[[ATTR0:[0-9]+]] { 
// CHECK: [[VECEXT:%.*]] = extractelement <4 x float> [[PADDR:%.*]], i32 1
// CHECK: [[VECEXT1:%.*]] = extractelement <4 x float> [[QADDR:%.*]], i32 1
// CHECK: [[MULRES:%.*]] = fmul {{[A-Za-z ]*}} float [[VECEXT]], [[VECEXT1]]
// CHECK: [[VECINIT:%.*]] = insertelement <4 x float> <float 1.000000e+00, float poison, float poison, float poison>, float [[MULRES]], i32 1
// CHECK: [[VECINIT2:%.*]] = extractelement <4 x float> [[PADDR2:%.*]], i32 2
// CHECK: [[VECINIT3:%.*]] = insertelement <4 x float> [[VECINIT]], float [[VECINIT2]], i32 2
// CHECK: [[VECINIT4:%.*]] = extractelement <4 x float> [[QADDR3:%.*]], i32 3
// CHECK: [[VECINIT5:%.*]] = insertelement <4 x float> [[VECINIT3]], float [[VECINIT4]], i32 3
// CHECK-NEXT: ret <4 x float> [[VECINIT5]]
float4 dstWithFloat(float4 p1, float4 p2)
{
    return dst(p1, p2);
}

// CHECK-LABEL: define {{.*}} <4 x half> @{{[A-Za-z1-9_]+}}dst_impl{{[A-Za-z1-9_]*}}(
// CHECK-SAME: <4 x half> {{[A-Za-z )(]*}} [[P:%.*]], <4 x half> {{[A-Za-z )(]*}} [[Q:%.*]]) #[[ATTR0]] {
// CHECK: [[VECEXT:%.*]] = extractelement <4 x half> [[PADDR:%.*]], i32 1
// CHECK: [[VECEXT1:%.*]] = extractelement <4 x half> [[QADDR:%.*]], i32 1
// CHECK: [[MULRES:%.*]] = fmul {{[A-Za-z ]*}} half [[VECEXT]], [[VECEXT1]]
// CHECK: [[VECINIT:%.*]] = insertelement <4 x half> <half 0xH3C00, half poison, half poison, half poison>, half [[MULRES]], i32 1
// CHECK: [[VECINIT2:%.*]] = extractelement <4 x half> [[PADDR2:%.*]], i32 2
// CHECK: [[VECINIT3:%.*]] = insertelement <4 x half> [[VECINIT]], half [[VECINIT2]], i32 2
// CHECK: [[VECINIT4:%.*]] = extractelement <4 x half> [[QADDR3:%.*]], i32 3
// CHECK: [[VECINIT5:%.*]] = insertelement <4 x half> [[VECINIT3]], half [[VECINIT4]], i32 3
// CHECK-NEXT: ret <4 x half> [[VECINIT5]]
half4 dstwithHalf(half4 p1, half4 p2)
{
    return dst(p1, p2);
}

// CHECK-LABEL: define {{.*}} <4 x double> @{{[A-Za-z1-9_]+}}dst_impl{{[A-Za-z1-9_]*}}(
// CHECK-SAME: <4 x double> {{[A-Za-z )(]*}} [[P:%.*]], <4 x double> {{[A-Za-z )(]*}} [[Q:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK: [[VECEXT:%.*]] = extractelement <4 x double> [[PADDR:%.*]], i32 1
// CHECK: [[VECEXT1:%.*]] = extractelement <4 x double> [[QADDR:%.*]], i32 1
// CHECK: [[MULRES:%.*]] = fmul {{[A-Za-z ]*}} double [[VECEXT]], [[VECEXT1]]
// CHECK: [[VECINIT:%.*]] = insertelement <4 x double> <double 1.000000e+00, double poison, double poison, double poison>, double [[MULRES]], i32 1
// CHECK: [[VECINIT2:%.*]] = extractelement <4 x double> [[PADDR2:%.*]], i32 2
// CHECK: [[VECINIT3:%.*]] = insertelement <4 x double> [[VECINIT]], double [[VECINIT2]], i32 2
// CHECK: [[VECINIT4:%.*]] = extractelement <4 x double> [[QADDR3:%.*]], i32 3
// CHECK: [[VECINIT5:%.*]] = insertelement <4 x double> [[VECINIT3]], double [[VECINIT4]], i32 3
// CHECK-NEXT: ret <4 x double> [[VECINIT5]]
double4 dstWithDouble(double4 p1, double4 p2)
{
    return dst(p1, p2);
}

