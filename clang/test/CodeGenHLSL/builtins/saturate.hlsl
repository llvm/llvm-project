// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF -Dtar=dx
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF -Dtar=dx

// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF -Dtar=spv
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF -Dtar=spv

// NATIVE_HALF-LABEL: define{{.*}} half @_Z18test_saturate_halfDh
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn half @llvm.[[tar]].saturate.f16(
// NO_HALF-LABEL: define{{.*}} float @_Z18test_saturate_halfDh
// NO_HALF: call reassoc nnan ninf nsz arcp afn float @llvm.[[tar]].saturate.f32(
half test_saturate_half(half p0) { return saturate(p0); }
// NATIVE_HALF-LABEL: define{{.*}} <2 x half> @_Z19test_saturate_half2Dv2_Dh
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.[[tar]].saturate.v2f16
// NO_HALF-LABEL: define{{.*}} <2 x float> @_Z19test_saturate_half2Dv2_Dh
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[tar]].saturate.v2f32(
half2 test_saturate_half2(half2 p0) { return saturate(p0); }
// NATIVE_HALF-LABEL: define{{.*}} <3 x half> @_Z19test_saturate_half3Dv3_Dh(
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.[[tar]].saturate.v3f16
// NO_HALF-LABEL: define{{.*}} <3 x float> @_Z19test_saturate_half3Dv3_Dh(<3 x float>
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[tar]].saturate.v3f32(
half3 test_saturate_half3(half3 p0) { return saturate(p0); }
// NATIVE_HALF-LABEL: define{{.*}} <4 x half> @_Z19test_saturate_half4Dv4_Dh(
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.[[tar]].saturate.v4f16
// NO_HALF-LABEL: define{{.*}} <4 x float> @_Z19test_saturate_half4Dv4_Dh(<4 x float>
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[tar]].saturate.v4f32(
half4 test_saturate_half4(half4 p0) { return saturate(p0); }

// CHECK-LABEL: define{{.*}} float @_Z19test_saturate_floatf(
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.[[tar]].saturate.f32(
float test_saturate_float(float p0) { return saturate(p0); }
// CHECK-LABEL: define{{.*}} <2 x float> @_Z20test_saturate_float2Dv2_f(<2 x float>
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[tar]].saturate.v2f32
float2 test_saturate_float2(float2 p0) { return saturate(p0); }
// CHECK-LABEL: define{{.*}} <3 x float> @_Z20test_saturate_float3Dv3_f(<3 x float>
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[tar]].saturate.v3f32
float3 test_saturate_float3(float3 p0) { return saturate(p0); }
// CHECK-LABEL: define{{.*}} <4 x float> @_Z20test_saturate_float4Dv4_f(<4 x float>
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[tar]].saturate.v4f32
float4 test_saturate_float4(float4 p0) { return saturate(p0); }

// CHECK-LABEL: define{{.*}} double @_Z20test_saturate_doubled(double
// CHECK: call reassoc nnan ninf nsz arcp afn double @llvm.[[tar]].saturate.f64(
double test_saturate_double(double p0) { return saturate(p0); }
// CHECK-LABEL: define{{.*}} <2 x double> @_Z21test_saturate_double2Dv2_d(<2 x double>
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.[[tar]].saturate.v2f64
double2 test_saturate_double2(double2 p0) { return saturate(p0); }
// CHECK-LABEL: define{{.*}} <3 x double> @_Z21test_saturate_double3Dv3_d(<3 x double>
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.[[tar]].saturate.v3f64
double3 test_saturate_double3(double3 p0) { return saturate(p0); }
// CHECK-LABEL: define{{.*}} <4 x double> @_Z21test_saturate_double4Dv4_d(<4 x double>
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.[[tar]].saturate.v4f64
double4 test_saturate_double4(double4 p0) { return saturate(p0); }
