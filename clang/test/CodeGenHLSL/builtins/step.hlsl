// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DFNATTRS=noundef -DTARGET=dx
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DFNATTRS=noundef -DTARGET=dx
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DFNATTRS="spir_func noundef" -DTARGET=spv
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DFNATTRS="spir_func noundef" -DTARGET=spv

// NATIVE_HALF: define [[FNATTRS]] half @
// NATIVE_HALF: call half @llvm.[[TARGET]].step.f16(half
// NO_HALF: call float @llvm.[[TARGET]].step.f32(float
// NATIVE_HALF: ret half
// NO_HALF: ret float
half test_step_half(half p0, half p1)
{
    return step(p0, p1);
}
// NATIVE_HALF: define [[FNATTRS]] <2 x half> @
// NATIVE_HALF: call <2 x half> @llvm.[[TARGET]].step.v2f16(<2 x half>
// NO_HALF: call <2 x float> @llvm.[[TARGET]].step.v2f32(<2 x float>
// NATIVE_HALF: ret <2 x half> %hlsl.step
// NO_HALF: ret <2 x float> %hlsl.step
half2 test_step_half2(half2 p0, half2 p1)
{
    return step(p0, p1);
}
// NATIVE_HALF: define [[FNATTRS]] <3 x half> @
// NATIVE_HALF: call <3 x half> @llvm.[[TARGET]].step.v3f16(<3 x half>
// NO_HALF: call <3 x float> @llvm.[[TARGET]].step.v3f32(<3 x float>
// NATIVE_HALF: ret <3 x half> %hlsl.step
// NO_HALF: ret <3 x float> %hlsl.step
half3 test_step_half3(half3 p0, half3 p1)
{
    return step(p0, p1);
}
// NATIVE_HALF: define [[FNATTRS]] <4 x half> @
// NATIVE_HALF: call <4 x half> @llvm.[[TARGET]].step.v4f16(<4 x half>
// NO_HALF: call <4 x float> @llvm.[[TARGET]].step.v4f32(<4 x float>
// NATIVE_HALF: ret <4 x half> %hlsl.step
// NO_HALF: ret <4 x float> %hlsl.step
half4 test_step_half4(half4 p0, half4 p1)
{
    return step(p0, p1);
}

// CHECK: define [[FNATTRS]] float @
// CHECK: call float @llvm.[[TARGET]].step.f32(float
// CHECK: ret float
float test_step_float(float p0, float p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.step = call <2 x float> @llvm.[[TARGET]].step.v2f32(
// CHECK: ret <2 x float> %hlsl.step
float2 test_step_float2(float2 p0, float2 p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.step = call <3 x float> @llvm.[[TARGET]].step.v3f32(
// CHECK: ret <3 x float> %hlsl.step
float3 test_step_float3(float3 p0, float3 p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.step = call <4 x float> @llvm.[[TARGET]].step.v4f32(
// CHECK: ret <4 x float> %hlsl.step
float4 test_step_float4(float4 p0, float4 p1)
{
    return step(p0, p1);
}
