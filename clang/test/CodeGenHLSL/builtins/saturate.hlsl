// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=SPIRV,SPIRV_HALF
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=SPIRV,SPIRV_NO_HALF

// NATIVE_HALF: define noundef half @
// NATIVE_HALF: call half @llvm.dx.saturate.f16(
// NO_HALF: define noundef float @"?test_saturate_half
// NO_HALF: call float @llvm.dx.saturate.f32(
// SPIRV_HALF: define spir_func noundef half @_Z18test_saturate_halfDh(half
// SPIRV_HALF: call half @llvm.spv.saturate.f16(half
// SPIRV_NO_HALF: define spir_func noundef float @_Z18test_saturate_halfDh(float
// SPIRV_NO_HALF: call float @llvm.spv.saturate.f32(float
half test_saturate_half(half p0) { return saturate(p0); }
// NATIVE_HALF: define noundef <2 x half> @
// NATIVE_HALF: call <2 x half> @llvm.dx.saturate.v2f16
// NO_HALF: define noundef <2 x float> @"?test_saturate_half2
// NO_HALF: call <2 x float> @llvm.dx.saturate.v2f32(
// SPIRV_HALF: define spir_func noundef <2 x half> @_Z19test_saturate_half2Dv2_Dh(
// SPIRV_HALF: call <2 x half> @llvm.spv.saturate.v2f16(<2 x half>
// SPIRV_NO_HALF: define spir_func noundef <2 x float> @_Z19test_saturate_half2Dv2_Dh(<2 x float>
// SPIRV_NO_HALF: call <2 x float> @llvm.spv.saturate.v2f32(<2 x float>
half2 test_saturate_half2(half2 p0) { return saturate(p0); }
// NATIVE_HALF: define noundef <3 x half> @
// NATIVE_HALF: call <3 x half> @llvm.dx.saturate.v3f16
// NO_HALF: define noundef <3 x float> @"?test_saturate_half3
// NO_HALF: call <3 x float> @llvm.dx.saturate.v3f32(
// SPIRV_HALF: define spir_func noundef <3 x half> @_Z19test_saturate_half3Dv3_Dh(
// SPIRV_HALF: call <3 x half> @llvm.spv.saturate.v3f16(<3 x half>
// SPIRV_NO_HALF: define spir_func noundef <3 x float> @_Z19test_saturate_half3Dv3_Dh(<3 x float>
// SPIRV_NO_HALF: call <3 x float> @llvm.spv.saturate.v3f32(<3 x float>
half3 test_saturate_half3(half3 p0) { return saturate(p0); }
// NATIVE_HALF: define noundef <4 x half> @
// NATIVE_HALF: call <4 x half> @llvm.dx.saturate.v4f16
// NO_HALF: define noundef <4 x float> @"?test_saturate_half4
// NO_HALF: call <4 x float> @llvm.dx.saturate.v4f32(
// SPIRV_HALF: define spir_func noundef <4 x half> @_Z19test_saturate_half4Dv4_Dh(
// SPIRV_HALF: call <4 x half> @llvm.spv.saturate.v4f16(<4 x half>
// SPIRV_NO_HALF: define spir_func noundef <4 x float> @_Z19test_saturate_half4Dv4_Dh(<4 x float>
// SPIRV_NO_HALF: call <4 x float> @llvm.spv.saturate.v4f32(<4 x float>
half4 test_saturate_half4(half4 p0) { return saturate(p0); }

// CHECK: define noundef float @"?test_saturate_float
// CHECK: call float @llvm.dx.saturate.f32(
// SPIRV: define spir_func noundef float @_Z19test_saturate_floatf(float
// SPIRV: call float @llvm.spv.saturate.f32(float
float test_saturate_float(float p0) { return saturate(p0); }
// CHECK: define noundef <2 x float> @"?test_saturate_float2
// CHECK: call <2 x float> @llvm.dx.saturate.v2f32
// SPIRV: define spir_func noundef <2 x float> @_Z20test_saturate_float2Dv2_f(<2 x float>
// SPIRV: call <2 x float> @llvm.spv.saturate.v2f32(<2 x float>
float2 test_saturate_float2(float2 p0) { return saturate(p0); }
// CHECK: define noundef <3 x float> @"?test_saturate_float3
// CHECK: call <3 x float> @llvm.dx.saturate.v3f32
// SPIRV: define spir_func noundef <3 x float> @_Z20test_saturate_float3Dv3_f(<3 x float>
// SPIRV: call <3 x float> @llvm.spv.saturate.v3f32(<3 x float>
float3 test_saturate_float3(float3 p0) { return saturate(p0); }
// CHECK: define noundef <4 x float> @"?test_saturate_float4
// CHECK: call <4 x float> @llvm.dx.saturate.v4f32
// SPIRV: define spir_func noundef <4 x float> @_Z20test_saturate_float4Dv4_f(<4 x float>
// SPIRV: call <4 x float> @llvm.spv.saturate.v4f32(<4 x float>
float4 test_saturate_float4(float4 p0) { return saturate(p0); }

// CHECK: define noundef double @
// CHECK: call double @llvm.dx.saturate.f64(
// SPIRV: define spir_func noundef double @_Z20test_saturate_doubled(double
// SPIRV: call double @llvm.spv.saturate.f64(double
double test_saturate_double(double p0) { return saturate(p0); }
// CHECK: define noundef <2 x double> @
// CHECK: call <2 x double> @llvm.dx.saturate.v2f64
// SPIRV: define spir_func noundef <2 x double> @_Z21test_saturate_double2Dv2_d(<2 x double>
// SPIRV: call <2 x double> @llvm.spv.saturate.v2f64(<2 x double>
double2 test_saturate_double2(double2 p0) { return saturate(p0); }
// CHECK: define noundef <3 x double> @
// CHECK: call <3 x double> @llvm.dx.saturate.v3f64
// SPIRV: define spir_func noundef <3 x double> @_Z21test_saturate_double3Dv3_d(<3 x double>
// SPIRV: call <3 x double> @llvm.spv.saturate.v3f64(<3 x double>
double3 test_saturate_double3(double3 p0) { return saturate(p0); }
// CHECK: define noundef <4 x double> @
// CHECK: call <4 x double> @llvm.dx.saturate.v4f64
// SPIRV: define spir_func noundef <4 x double> @_Z21test_saturate_double4Dv4_d(<4 x double>
// SPIRV: call <4 x double> @llvm.spv.saturate.v4f64(<4 x double>
double4 test_saturate_double4(double4 p0) { return saturate(p0); }
