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
// DXIL_NATIVE_HALF: call half @llvm.dx.normalize.f16(half
// SPIR_NATIVE_HALF: call half @llvm.spv.normalize.f16(half
// DXIL_NO_HALF: call float @llvm.dx.normalize.f32(float
// SPIR_NO_HALF: call float @llvm.spv.normalize.f32(float
// NATIVE_HALF: ret half
// NO_HALF: ret float
half test_normalize_half(half p0)
{
    return normalize(p0);
}
// DXIL_NATIVE_HALF: define noundef <2 x half> @
// SPIR_NATIVE_HALF: define spir_func noundef <2 x half> @
// DXIL_NATIVE_HALF: call <2 x half> @llvm.dx.normalize.v2f16(<2 x half>
// SPIR_NATIVE_HALF: call <2 x half> @llvm.spv.normalize.v2f16(<2 x half>
// DXIL_NO_HALF: call <2 x float> @llvm.dx.normalize.v2f32(<2 x float>
// SPIR_NO_HALF: call <2 x float> @llvm.spv.normalize.v2f32(<2 x float>
// NATIVE_HALF: ret <2 x half> %hlsl.normalize
// NO_HALF: ret <2 x float> %hlsl.normalize
half2 test_normalize_half2(half2 p0)
{
    return normalize(p0);
}
// DXIL_NATIVE_HALF: define noundef <3 x half> @
// SPIR_NATIVE_HALF: define spir_func noundef <3 x half> @
// DXIL_NATIVE_HALF: call <3 x half> @llvm.dx.normalize.v3f16(<3 x half>
// SPIR_NATIVE_HALF: call <3 x half> @llvm.spv.normalize.v3f16(<3 x half>
// DXIL_NO_HALF: call <3 x float> @llvm.dx.normalize.v3f32(<3 x float>
// SPIR_NO_HALF: call <3 x float> @llvm.spv.normalize.v3f32(<3 x float>
// NATIVE_HALF: ret <3 x half> %hlsl.normalize
// NO_HALF: ret <3 x float> %hlsl.normalize
half3 test_normalize_half3(half3 p0)
{
    return normalize(p0);
}
// DXIL_NATIVE_HALF: define noundef <4 x half> @
// SPIR_NATIVE_HALF: define spir_func noundef <4 x half> @
// DXIL_NATIVE_HALF: call <4 x half> @llvm.dx.normalize.v4f16(<4 x half>
// SPIR_NATIVE_HALF: call <4 x half> @llvm.spv.normalize.v4f16(<4 x half>
// DXIL_NO_HALF: call <4 x float> @llvm.dx.normalize.v4f32(<4 x float>
// SPIR_NO_HALF: call <4 x float> @llvm.spv.normalize.v4f32(<4 x float>
// NATIVE_HALF: ret <4 x half> %hlsl.normalize
// NO_HALF: ret <4 x float> %hlsl.normalize
half4 test_normalize_half4(half4 p0)
{
    return normalize(p0);
}

// DXIL_CHECK: define noundef float @
// SPIR_CHECK: define spir_func noundef float @
// DXIL_CHECK: call float @llvm.dx.normalize.f32(float
// SPIR_CHECK: call float @llvm.spv.normalize.f32(float
// CHECK: ret float
float test_normalize_float(float p0)
{
    return normalize(p0);
}
// DXIL_CHECK: define noundef <2 x float> @
// SPIR_CHECK: define spir_func noundef <2 x float> @
// DXIL_CHECK: %hlsl.normalize = call <2 x float> @llvm.dx.normalize.v2f32(
// SPIR_CHECK: %hlsl.normalize = call <2 x float> @llvm.spv.normalize.v2f32(<2 x float>
// CHECK: ret <2 x float> %hlsl.normalize
float2 test_normalize_float2(float2 p0)
{
    return normalize(p0);
}
// DXIL_CHECK: define noundef <3 x float> @
// SPIR_CHECK: define spir_func noundef <3 x float> @
// DXIL_CHECK: %hlsl.normalize = call <3 x float> @llvm.dx.normalize.v3f32(
// SPIR_CHECK: %hlsl.normalize = call <3 x float> @llvm.spv.normalize.v3f32(<3 x float>
// CHECK: ret <3 x float> %hlsl.normalize
float3 test_normalize_float3(float3 p0)
{
    return normalize(p0);
}
// DXIL_CHECK: define noundef <4 x float> @
// SPIR_CHECK: define spir_func noundef <4 x float> @
// DXIL_CHECK: %hlsl.normalize = call <4 x float> @llvm.dx.normalize.v4f32(
// SPIR_CHECK: %hlsl.normalize = call <4 x float> @llvm.spv.normalize.v4f32(
// CHECK: ret <4 x float> %hlsl.normalize
float4 test_length_float4(float4 p0)
{
    return normalize(p0);
}
