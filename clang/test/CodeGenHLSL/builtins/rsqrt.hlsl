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
// NATIVE_HALF: %hlsl.rsqrt = call half @llvm.[[TARGET]].rsqrt.f16(
// NATIVE_HALF: ret half %hlsl.rsqrt
// NO_HALF: define [[FNATTRS]] float @
// NO_HALF: %hlsl.rsqrt = call float @llvm.[[TARGET]].rsqrt.f32(
// NO_HALF: ret float %hlsl.rsqrt
half test_rsqrt_half(half p0) { return rsqrt(p0); }
// NATIVE_HALF: define [[FNATTRS]] <2 x half> @
// NATIVE_HALF: %hlsl.rsqrt = call <2 x half> @llvm.[[TARGET]].rsqrt.v2f16
// NATIVE_HALF: ret <2 x half> %hlsl.rsqrt
// NO_HALF: define [[FNATTRS]] <2 x float> @
// NO_HALF: %hlsl.rsqrt = call <2 x float> @llvm.[[TARGET]].rsqrt.v2f32(
// NO_HALF: ret <2 x float> %hlsl.rsqrt
half2 test_rsqrt_half2(half2 p0) { return rsqrt(p0); }
// NATIVE_HALF: define [[FNATTRS]] <3 x half> @
// NATIVE_HALF: %hlsl.rsqrt = call <3 x half> @llvm.[[TARGET]].rsqrt.v3f16
// NATIVE_HALF: ret <3 x half> %hlsl.rsqrt
// NO_HALF: define [[FNATTRS]] <3 x float> @
// NO_HALF: %hlsl.rsqrt = call <3 x float> @llvm.[[TARGET]].rsqrt.v3f32(
// NO_HALF: ret <3 x float> %hlsl.rsqrt
half3 test_rsqrt_half3(half3 p0) { return rsqrt(p0); }
// NATIVE_HALF: define [[FNATTRS]] <4 x half> @
// NATIVE_HALF: %hlsl.rsqrt = call <4 x half> @llvm.[[TARGET]].rsqrt.v4f16
// NATIVE_HALF: ret <4 x half> %hlsl.rsqrt
// NO_HALF: define [[FNATTRS]] <4 x float> @
// NO_HALF: %hlsl.rsqrt = call <4 x float> @llvm.[[TARGET]].rsqrt.v4f32(
// NO_HALF: ret <4 x float> %hlsl.rsqrt
half4 test_rsqrt_half4(half4 p0) { return rsqrt(p0); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %hlsl.rsqrt = call float @llvm.[[TARGET]].rsqrt.f32(
// CHECK: ret float %hlsl.rsqrt
float test_rsqrt_float(float p0) { return rsqrt(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.rsqrt = call <2 x float> @llvm.[[TARGET]].rsqrt.v2f32
// CHECK: ret <2 x float> %hlsl.rsqrt
float2 test_rsqrt_float2(float2 p0) { return rsqrt(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.rsqrt = call <3 x float> @llvm.[[TARGET]].rsqrt.v3f32
// CHECK: ret <3 x float> %hlsl.rsqrt
float3 test_rsqrt_float3(float3 p0) { return rsqrt(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.rsqrt = call <4 x float> @llvm.[[TARGET]].rsqrt.v4f32
// CHECK: ret <4 x float> %hlsl.rsqrt
float4 test_rsqrt_float4(float4 p0) { return rsqrt(p0); }
