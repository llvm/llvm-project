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
// RUN:   --check-prefixes=CHECK,NATIVE_HALF,SPIR_NATIVE_HALF,SPIR_CHECK
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF,SPIR_NO_HALF,SPIR_CHECK

// DXIL_NATIVE_HALF: define noundef half @
// SPIR_NATIVE_HALF: define spir_func noundef half @
// DXIL_NATIVE_HALF: call half @llvm.dx.step.f16(half
// SPIR_NATIVE_HALF: call half @llvm.spv.step.f16(half
// DXIL_NO_HALF: call float @llvm.dx.step.f32(float
// SPIR_NO_HALF: call float @llvm.spv.step.f32(float
// NATIVE_HALF: ret half
// NO_HALF: ret float
half test_step_half(half p0, half p1)
{
    return step(p0, p1);
}
// DXIL_NATIVE_HALF: define noundef <2 x half> @
// SPIR_NATIVE_HALF: define spir_func noundef <2 x half> @
// DXIL_NATIVE_HALF: call <2 x half> @llvm.dx.step.v2f16(<2 x half>
// SPIR_NATIVE_HALF: call <2 x half> @llvm.spv.step.v2f16(<2 x half>
// DXIL_NO_HALF: call <2 x float> @llvm.dx.step.v2f32(<2 x float>
// SPIR_NO_HALF: call <2 x float> @llvm.spv.step.v2f32(<2 x float>
// NATIVE_HALF: ret <2 x half> %hlsl.step
// NO_HALF: ret <2 x float> %hlsl.step
half2 test_step_half2(half2 p0, half2 p1)
{
    return step(p0, p1);
}
// DXIL_NATIVE_HALF: define noundef <3 x half> @
// SPIR_NATIVE_HALF: define spir_func noundef <3 x half> @
// DXIL_NATIVE_HALF: call <3 x half> @llvm.dx.step.v3f16(<3 x half>
// SPIR_NATIVE_HALF: call <3 x half> @llvm.spv.step.v3f16(<3 x half>
// DXIL_NO_HALF: call <3 x float> @llvm.dx.step.v3f32(<3 x float>
// SPIR_NO_HALF: call <3 x float> @llvm.spv.step.v3f32(<3 x float>
// NATIVE_HALF: ret <3 x half> %hlsl.step
// NO_HALF: ret <3 x float> %hlsl.step
half3 test_step_half3(half3 p0, half3 p1)
{
    return step(p0, p1);
}
// DXIL_NATIVE_HALF: define noundef <4 x half> @
// SPIR_NATIVE_HALF: define spir_func noundef <4 x half> @
// DXIL_NATIVE_HALF: call <4 x half> @llvm.dx.step.v4f16(<4 x half>
// SPIR_NATIVE_HALF: call <4 x half> @llvm.spv.step.v4f16(<4 x half>
// DXIL_NO_HALF: call <4 x float> @llvm.dx.step.v4f32(<4 x float>
// SPIR_NO_HALF: call <4 x float> @llvm.spv.step.v4f32(<4 x float>
// NATIVE_HALF: ret <4 x half> %hlsl.step
// NO_HALF: ret <4 x float> %hlsl.step
half4 test_step_half4(half4 p0, half4 p1)
{
    return step(p0, p1);
}

// DXIL_CHECK: define noundef float @
// SPIR_CHECK: define spir_func noundef float @
// DXIL_CHECK: call float @llvm.dx.step.f32(float
// SPIR_CHECK: call float @llvm.spv.step.f32(float
// CHECK: ret float
float test_step_float(float p0, float p1)
{
    return step(p0, p1);
}
// DXIL_CHECK: define noundef <2 x float> @
// SPIR_CHECK: define spir_func noundef <2 x float> @
// DXIL_CHECK: %hlsl.step = call <2 x float> @llvm.dx.step.v2f32(
// SPIR_CHECK: %hlsl.step = call <2 x float> @llvm.spv.step.v2f32(<2 x float>
// CHECK: ret <2 x float> %hlsl.step
float2 test_step_float2(float2 p0, float2 p1)
{
    return step(p0, p1);
}
// DXIL_CHECK: define noundef <3 x float> @
// SPIR_CHECK: define spir_func noundef <3 x float> @
// DXIL_CHECK: %hlsl.step = call <3 x float> @llvm.dx.step.v3f32(
// SPIR_CHECK: %hlsl.step = call <3 x float> @llvm.spv.step.v3f32(<3 x float>
// CHECK: ret <3 x float> %hlsl.step
float3 test_step_float3(float3 p0, float3 p1)
{
    return step(p0, p1);
}
// DXIL_CHECK: define noundef <4 x float> @
// SPIR_CHECK: define spir_func noundef <4 x float> @
// DXIL_CHECK: %hlsl.step = call <4 x float> @llvm.dx.step.v4f32(
// SPIR_CHECK: %hlsl.step = call <4 x float> @llvm.spv.step.v4f32(
// CHECK: ret <4 x float> %hlsl.step
float4 test_step_float4(float4 p0, float4 p1)
{
    return step(p0, p1);
}