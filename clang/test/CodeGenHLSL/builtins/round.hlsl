// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF-LABEL: define hidden noundef nofpclass(nan inf) half @_Z15test_round_half
// NATIVE_HALF: [[ROUNDEVEN:%.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.roundeven.f16(
// NATIVE_HALF: ret half [[ROUNDEVEN]]
// NO_HALF-LABEL: define hidden noundef nofpclass(nan inf) float @_Z15test_round_half
// NO_HALF: [[ROUNDEVEN:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.roundeven.f32(
// NO_HALF: ret float [[ROUNDEVEN]]
half test_round_half(half p0) { return round(p0); }
// NATIVE_HALF-LABEL: define hidden noundef nofpclass(nan inf) <2 x half> @_Z16test_round_half2
// NATIVE_HALF: [[ROUNDEVEN:%.*]] = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.roundeven.v2f16
// NATIVE_HALF: ret <2 x half> [[ROUNDEVEN:%.*]]
// NO_HALF-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> @_Z16test_round_half2
// NO_HALF: [[ROUNDEVEN:%.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.roundeven.v2f32(
// NO_HALF: ret <2 x float> [[ROUNDEVEN]]
half2 test_round_half2(half2 p0) { return round(p0); }
// NATIVE_HALF-LABEL: define hidden noundef nofpclass(nan inf) <3 x half> @_Z16test_round_half3
// NATIVE_HALF: [[ROUNDEVEN:%.*]] = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.roundeven.v3f16
// NATIVE_HALF: ret <3 x half> [[ROUNDEVEN:%.*]]
// NO_HALF-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> @_Z16test_round_half3
// NO_HALF: [[ROUNDEVEN:%.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.roundeven.v3f32(
// NO_HALF: ret <3 x float> [[ROUNDEVEN]]
half3 test_round_half3(half3 p0) { return round(p0); }
// NATIVE_HALF-LABEL: define hidden noundef nofpclass(nan inf) <4 x half> @_Z16test_round_half4
// NATIVE_HALF: [[ROUNDEVEN:%.*]] = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.roundeven.v4f16
// NATIVE_HALF: ret <4 x half> [[ROUNDEVEN:%.*]]
// NO_HALF-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> @_Z16test_round_half4
// NO_HALF: [[ROUNDEVEN:%.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.roundeven.v4f32(
// NO_HALF: ret <4 x float> [[ROUNDEVEN]]
half4 test_round_half4(half4 p0) { return round(p0); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float @_Z16test_round_float
// CHECK: [[ROUNDEVEN:%.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.roundeven.f32(
// CHECK: ret float [[ROUNDEVEN]]
float test_round_float(float p0) { return round(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> @_Z17test_round_float2
// CHECK: [[ROUNDEVEN:%.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.roundeven.v2f32
// CHECK: ret <2 x float> [[ROUNDEVEN]]
float2 test_round_float2(float2 p0) { return round(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> @_Z17test_round_float3
// CHECK: [[ROUNDEVEN:%.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.roundeven.v3f32
// CHECK: ret <3 x float> [[ROUNDEVEN]]
float3 test_round_float3(float3 p0) { return round(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> @_Z17test_round_float4
// CHECK: [[ROUNDEVEN:%.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.roundeven.v4f32
// CHECK: ret <4 x float> [[ROUNDEVEN]]
float4 test_round_float4(float4 p0) { return round(p0); }
