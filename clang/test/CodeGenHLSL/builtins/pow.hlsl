// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) half @_Z13test_pow_half
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn half @llvm.pow.f16(
// NO_HALF-LABEL: define noundef nofpclass(nan inf) float @_Z13test_pow_half
// NO_HALF: call reassoc nnan ninf nsz arcp afn float @llvm.pow.f32(
half test_pow_half(half p0, half p1) { return pow(p0, p1); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <2 x half> @_Z14test_pow_half2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.pow.v2f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z14test_pow_half2
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.pow.v2f32(
half2 test_pow_half2(half2 p0, half2 p1) { return pow(p0, p1); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <3 x half> @_Z14test_pow_half3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.pow.v3f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z14test_pow_half3
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.pow.v3f32(
half3 test_pow_half3(half3 p0, half3 p1) { return pow(p0, p1); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <4 x half> @_Z14test_pow_half4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.pow.v4f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z14test_pow_half4
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.pow.v4f32(
half4 test_pow_half4(half4 p0, half4 p1) { return pow(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float @_Z14test_pow_float
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.pow.f32(
float test_pow_float(float p0, float p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z15test_pow_float2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.pow.v2f32
float2 test_pow_float2(float2 p0, float2 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z15test_pow_float3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.pow.v3f32
float3 test_pow_float3(float3 p0, float3 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z15test_pow_float4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.pow.v4f32
float4 test_pow_float4(float4 p0, float4 p1) { return pow(p0, p1); }
