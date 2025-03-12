// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) half @_Z14test_sqrt_half
// NATIVE_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn half @llvm.sqrt.f16(
// NATIVE_HALF: ret half %{{.*}}
// NO_HALF-LABEL: define noundef nofpclass(nan inf) float @_Z14test_sqrt_half
// NO_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.sqrt.f32(
// NO_HALF: ret float %{{.*}}
half test_sqrt_half(half p0) { return sqrt(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <2 x half> @_Z15test_sqrt_half2
// NATIVE_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.sqrt.v2f16
// NATIVE_HALF: ret <2 x half> %{{.*}}
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z15test_sqrt_half2
// NO_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sqrt.v2f32(
// NO_HALF: ret <2 x float> %{{.*}}
half2 test_sqrt_half2(half2 p0) { return sqrt(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <3 x half> @_Z15test_sqrt_half3
// NATIVE_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.sqrt.v3f16
// NATIVE_HALF: ret <3 x half> %{{.*}}
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z15test_sqrt_half3
// NO_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sqrt.v3f32(
// NO_HALF: ret <3 x float> %{{.*}}
half3 test_sqrt_half3(half3 p0) { return sqrt(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <4 x half> @_Z15test_sqrt_half4
// NATIVE_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.sqrt.v4f16
// NATIVE_HALF: ret <4 x half> %{{.*}}
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z15test_sqrt_half4
// NO_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sqrt.v4f32(
// NO_HALF: ret <4 x float> %{{.*}}
half4 test_sqrt_half4(half4 p0) { return sqrt(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float @_Z15test_sqrt_float
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.sqrt.f32(
// CHECK: ret float %{{.*}}
float test_sqrt_float(float p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z16test_sqrt_float2
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sqrt.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_sqrt_float2(float2 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z16test_sqrt_float3
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sqrt.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_sqrt_float3(float3 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z16test_sqrt_float4
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sqrt.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_sqrt_float4(float4 p0) { return sqrt(p0); }
