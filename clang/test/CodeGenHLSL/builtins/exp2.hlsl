// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) half @_Z14test_exp2_half
// NATIVE_HALF: %elt.exp2 = call reassoc nnan ninf nsz arcp afn half @llvm.exp2.f16(
// NATIVE_HALF: ret half %elt.exp2
// NO_HALF-LABEL: define noundef nofpclass(nan inf) float @_Z14test_exp2_half
// NO_HALF: %elt.exp2 = call reassoc nnan ninf nsz arcp afn float @llvm.exp2.f32(
// NO_HALF: ret float %elt.exp2
half test_exp2_half(half p0) { return exp2(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <2 x half> @_Z15test_exp2_half2
// NATIVE_HALF: %elt.exp2 = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.exp2.v2f16
// NATIVE_HALF: ret <2 x half> %elt.exp2
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z15test_exp2_half2
// NO_HALF: %elt.exp2 = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.exp2.v2f32(
// NO_HALF: ret <2 x float> %elt.exp2
half2 test_exp2_half2(half2 p0) { return exp2(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <3 x half> @_Z15test_exp2_half3
// NATIVE_HALF: %elt.exp2 = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.exp2.v3f16
// NATIVE_HALF: ret <3 x half> %elt.exp2
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z15test_exp2_half3
// NO_HALF: %elt.exp2 = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.exp2.v3f32(
// NO_HALF: ret <3 x float> %elt.exp2
half3 test_exp2_half3(half3 p0) { return exp2(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <4 x half> @_Z15test_exp2_half4
// NATIVE_HALF: %elt.exp2 = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.exp2.v4f16
// NATIVE_HALF: ret <4 x half> %elt.exp2
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z15test_exp2_half4
// NO_HALF: %elt.exp2 = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.exp2.v4f32(
// NO_HALF: ret <4 x float> %elt.exp2
half4 test_exp2_half4(half4 p0) { return exp2(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float @_Z15test_exp2_float
// CHECK: %elt.exp2 = call reassoc nnan ninf nsz arcp afn float @llvm.exp2.f32(
// CHECK: ret float %elt.exp2
float test_exp2_float(float p0) { return exp2(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z16test_exp2_float2
// CHECK: %elt.exp2 = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.exp2.v2f32
// CHECK: ret <2 x float> %elt.exp2
float2 test_exp2_float2(float2 p0) { return exp2(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z16test_exp2_float3
// CHECK: %elt.exp2 = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.exp2.v3f32
// CHECK: ret <3 x float> %elt.exp2
float3 test_exp2_float3(float3 p0) { return exp2(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z16test_exp2_float4
// CHECK: %elt.exp2 = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.exp2.v4f32
// CHECK: ret <4 x float> %elt.exp2
float4 test_exp2_float4(float4 p0) { return exp2(p0); }
