// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) half @_Z13test_cos_half
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn half @llvm.cos.f16(
// NO_HALF-LABEL: define noundef nofpclass(nan inf) float @_Z13test_cos_half
// NO_HALF: call reassoc nnan ninf nsz arcp afn float @llvm.cos.f32(
half test_cos_half(half p0) { return cos(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <2 x half> @_Z14test_cos_half2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.cos.v2f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z14test_cos_half2
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.cos.v2f32(
half2 test_cos_half2(half2 p0) { return cos(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <3 x half> @_Z14test_cos_half3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.cos.v3f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z14test_cos_half3
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.cos.v3f32(
half3 test_cos_half3(half3 p0) { return cos(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <4 x half> @_Z14test_cos_half4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.cos.v4f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z14test_cos_half4
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.cos.v4f32(
half4 test_cos_half4(half4 p0) { return cos(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float @_Z14test_cos_float
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.cos.f32(
float test_cos_float(float p0) { return cos(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z15test_cos_float2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.cos.v2f32
float2 test_cos_float2(float2 p0) { return cos(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z15test_cos_float3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.cos.v3f32
float3 test_cos_float3(float3 p0) { return cos(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z15test_cos_float4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.cos.v4f32
float4 test_cos_float4(float4 p0) { return cos(p0); }
