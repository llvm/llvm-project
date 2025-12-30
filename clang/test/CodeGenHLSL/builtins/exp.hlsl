// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF-LABEL: define hidden noundef nofpclass(nan inf) half @_Z13test_exp_half
// NATIVE_HALF: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.exp.f16(
// NATIVE_HALF: ret half [[EXP]]
// NO_HALF-LABEL: define hidden noundef nofpclass(nan inf) float @_Z13test_exp_half
// NO_HALF: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(
// NO_HALF: ret float [[EXP]]
half test_exp_half(half p0) { return exp(p0); }
// NATIVE_HALF-LABEL: define hidden noundef nofpclass(nan inf) <2 x half> @_Z14test_exp_half2
// NATIVE_HALF: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.exp.v2f16
// NATIVE_HALF: ret <2 x half> [[EXP]]
// NO_HALF-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> @_Z14test_exp_half2
// NO_HALF: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.exp.v2f32(
// NO_HALF: ret <2 x float> [[EXP]]
half2 test_exp_half2(half2 p0) { return exp(p0); }
// NATIVE_HALF-LABEL: define hidden noundef nofpclass(nan inf) <3 x half> @_Z14test_exp_half3
// NATIVE_HALF: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.exp.v3f16
// NATIVE_HALF: ret <3 x half> [[EXP]]
// NO_HALF-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> @_Z14test_exp_half3
// NO_HALF: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.exp.v3f32(
// NO_HALF: ret <3 x float> [[EXP]]
half3 test_exp_half3(half3 p0) { return exp(p0); }
// NATIVE_HALF-LABEL: define hidden noundef nofpclass(nan inf) <4 x half> @_Z14test_exp_half4
// NATIVE_HALF: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.exp.v4f16
// NATIVE_HALF: ret <4 x half> [[EXP]]
// NO_HALF-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> @_Z14test_exp_half4
// NO_HALF: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.exp.v4f32(
// NO_HALF: ret <4 x float> [[EXP]]
half4 test_exp_half4(half4 p0) { return exp(p0); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float @_Z14test_exp_float
// CHECK: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(
// CHECK: ret float [[EXP:%.*]]
float test_exp_float(float p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> @_Z15test_exp_float2
// CHECK: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.exp.v2f32
// CHECK: ret <2 x float> [[EXP]]
float2 test_exp_float2(float2 p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> @_Z15test_exp_float3
// CHECK: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.exp.v3f32
// CHECK: ret <3 x float> [[EXP]]
float3 test_exp_float3(float3 p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> @_Z15test_exp_float4
// CHECK: [[EXP:%.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.exp.v4f32
// CHECK: ret <4 x float> [[EXP]]
float4 test_exp_float4(float4 p0) { return exp(p0); }
