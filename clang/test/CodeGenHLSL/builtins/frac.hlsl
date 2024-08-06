// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,DXIL_CHECK,DXIL_NATIVE_HALF,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,DXIL_CHECK,NO_HALF,DXIL_NO_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SPIR_CHECK,NATIVE_HALF,SPIR_NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,SPIR_CHECK,NO_HALF,SPIR_NO_HALF

// DXIL_NATIVE_HALF: define noundef half @
// SPIR_NATIVE_HALF: define spir_func noundef half @
// DXIL_NATIVE_HALF: %hlsl.frac = call half @llvm.dx.frac.f16(
// SPIR_NATIVE_HALF: %hlsl.frac = call half @llvm.spv.frac.f16(
// NATIVE_HALF: ret half %hlsl.frac
// DXIL_NO_HALF: define noundef float @
// SPIR_NO_HALF: define spir_func noundef float @
// DXIL_NO_HALF: %hlsl.frac = call float @llvm.dx.frac.f32(
// SPIR_NO_HALF: %hlsl.frac = call float @llvm.spv.frac.f32(
// NO_HALF: ret float %hlsl.frac
half test_frac_half(half p0) { return frac(p0); }
// DXIL_NATIVE_HALF: define noundef <2 x half> @
// SPIR_NATIVE_HALF: define spir_func noundef <2 x half> @
// DXIL_NATIVE_HALF: %hlsl.frac = call <2 x half> @llvm.dx.frac.v2f16
// SPIR_NATIVE_HALF: %hlsl.frac = call <2 x half> @llvm.spv.frac.v2f16
// NATIVE_HALF: ret <2 x half> %hlsl.frac
// DXIL_NO_HALF: define noundef <2 x float> @
// SPIR_NO_HALF: define spir_func noundef <2 x float> @
// DXIL_NO_HALF: %hlsl.frac = call <2 x float> @llvm.dx.frac.v2f32(
// SPIR_NO_HALF: %hlsl.frac = call <2 x float> @llvm.spv.frac.v2f32(
// NO_HALF: ret <2 x float> %hlsl.frac
half2 test_frac_half2(half2 p0) { return frac(p0); }
// DXIL_NATIVE_HALF: define noundef <3 x half> @
// SPIR_NATIVE_HALF: define spir_func noundef <3 x half> @
// DXIL_NATIVE_HALF: %hlsl.frac = call <3 x half> @llvm.dx.frac.v3f16
// SPIR_NATIVE_HALF: %hlsl.frac = call <3 x half> @llvm.spv.frac.v3f16
// NATIVE_HALF: ret <3 x half> %hlsl.frac
// DXIL_NO_HALF: define noundef <3 x float> @
// SPIR_NO_HALF: define spir_func noundef <3 x float> @
// DXIL_NO_HALF: %hlsl.frac = call <3 x float> @llvm.dx.frac.v3f32(
// SPIR_NO_HALF: %hlsl.frac = call <3 x float> @llvm.spv.frac.v3f32(
// NO_HALF: ret <3 x float> %hlsl.frac
half3 test_frac_half3(half3 p0) { return frac(p0); }
// DXIL_NATIVE_HALF: define noundef <4 x half> @
// SPIR_NATIVE_HALF: define spir_func noundef <4 x half> @
// DXIL_NATIVE_HALF: %hlsl.frac = call <4 x half> @llvm.dx.frac.v4f16
// SPIR_NATIVE_HALF: %hlsl.frac = call <4 x half> @llvm.spv.frac.v4f16
// NATIVE_HALF: ret <4 x half> %hlsl.frac
// DXIL_NO_HALF: define noundef <4 x float> @
// SPIR_NO_HALF: define spir_func noundef <4 x float> @
// DXIL_NO_HALF: %hlsl.frac = call <4 x float> @llvm.dx.frac.v4f32(
// SPIR_NO_HALF: %hlsl.frac = call <4 x float> @llvm.spv.frac.v4f32(
// NO_HALF: ret <4 x float> %hlsl.frac
half4 test_frac_half4(half4 p0) { return frac(p0); }

// DXIL_CHECK: define noundef float @
// SPIR_CHECK: define spir_func noundef float @
// DXIL_CHECK: %hlsl.frac = call float @llvm.dx.frac.f32(
// SPIR_CHECK: %hlsl.frac = call float @llvm.spv.frac.f32(
// CHECK: ret float %hlsl.frac
float test_frac_float(float p0) { return frac(p0); }
// DXIL_CHECK: define noundef <2 x float> @
// SPIR_CHECK: define spir_func noundef <2 x float> @
// DXIL_CHECK: %hlsl.frac = call <2 x float> @llvm.dx.frac.v2f32
// SPIR_CHECK: %hlsl.frac = call <2 x float> @llvm.spv.frac.v2f32
// CHECK: ret <2 x float> %hlsl.frac
float2 test_frac_float2(float2 p0) { return frac(p0); }
// DXIL_CHECK: define noundef <3 x float> @
// SPIR_CHECK: define spir_func noundef <3 x float> @
// DXIL_CHECK: %hlsl.frac = call <3 x float> @llvm.dx.frac.v3f32
// SPIR_CHECK: %hlsl.frac = call <3 x float> @llvm.spv.frac.v3f32
// CHECK: ret <3 x float> %hlsl.frac
float3 test_frac_float3(float3 p0) { return frac(p0); }
// DXIL_CHECK: define noundef <4 x float> @
// SPIR_CHECK: define spir_func noundef <4 x float> @
// DXIL_CHECK: %hlsl.frac = call <4 x float> @llvm.dx.frac.v4f32
// SPIR_CHECK: %hlsl.frac = call <4 x float> @llvm.spv.frac.v4f32
// CHECK: ret <4 x float> %hlsl.frac
float4 test_frac_float4(float4 p0) { return frac(p0); }
