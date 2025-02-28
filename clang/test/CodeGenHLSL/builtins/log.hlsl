// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) half @_Z13test_log_half
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn half @llvm.log.f16(
// NO_HALF-LABEL: define noundef nofpclass(nan inf) float @_Z13test_log_half
// NO_HALF: call reassoc nnan ninf nsz arcp afn float @llvm.log.f32(
half test_log_half(half p0) { return log(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <2 x half> @_Z14test_log_half2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.log.v2f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z14test_log_half2
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.log.v2f32(
half2 test_log_half2(half2 p0) { return log(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <3 x half> @_Z14test_log_half3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.log.v3f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z14test_log_half3
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.log.v3f32(
half3 test_log_half3(half3 p0) { return log(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <4 x half> @_Z14test_log_half4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.log.v4f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z14test_log_half4
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.log.v4f32(
half4 test_log_half4(half4 p0) { return log(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float @_Z14test_log_float
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.log.f32(
float test_log_float(float p0) { return log(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z15test_log_float2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.log.v2f32
float2 test_log_float2(float2 p0) { return log(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z15test_log_float3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.log.v3f32
float3 test_log_float3(float3 p0) { return log(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z15test_log_float4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.log.v4f32
float4 test_log_float4(float4 p0) { return log(p0); }
