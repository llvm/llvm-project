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
// NATIVE_HALF: %hlsl.frac = call half @llvm.[[TARGET]].frac.f16(
// NATIVE_HALF: ret half %hlsl.frac
// NO_HALF: define [[FNATTRS]] float @
// NO_HALF: %hlsl.frac = call float @llvm.[[TARGET]].frac.f32(
// NO_HALF: ret float %hlsl.frac
half test_frac_half(half p0) { return frac(p0); }
// NATIVE_HALF: define [[FNATTRS]] <2 x half> @
// NATIVE_HALF: %hlsl.frac = call <2 x half> @llvm.[[TARGET]].frac.v2f16
// NATIVE_HALF: ret <2 x half> %hlsl.frac
// NO_HALF: define [[FNATTRS]] <2 x float> @
// NO_HALF: %hlsl.frac = call <2 x float> @llvm.[[TARGET]].frac.v2f32(
// NO_HALF: ret <2 x float> %hlsl.frac
half2 test_frac_half2(half2 p0) { return frac(p0); }
// NATIVE_HALF: define [[FNATTRS]] <3 x half> @
// NATIVE_HALF: %hlsl.frac = call <3 x half> @llvm.[[TARGET]].frac.v3f16
// NATIVE_HALF: ret <3 x half> %hlsl.frac
// NO_HALF: define [[FNATTRS]] <3 x float> @
// NO_HALF: %hlsl.frac = call <3 x float> @llvm.[[TARGET]].frac.v3f32(
// NO_HALF: ret <3 x float> %hlsl.frac
half3 test_frac_half3(half3 p0) { return frac(p0); }
// NATIVE_HALF: define [[FNATTRS]] <4 x half> @
// NATIVE_HALF: %hlsl.frac = call <4 x half> @llvm.[[TARGET]].frac.v4f16
// NATIVE_HALF: ret <4 x half> %hlsl.frac
// NO_HALF: define [[FNATTRS]] <4 x float> @
// NO_HALF: %hlsl.frac = call <4 x float> @llvm.[[TARGET]].frac.v4f32(
// NO_HALF: ret <4 x float> %hlsl.frac
half4 test_frac_half4(half4 p0) { return frac(p0); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %hlsl.frac = call float @llvm.[[TARGET]].frac.f32(
// CHECK: ret float %hlsl.frac
float test_frac_float(float p0) { return frac(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.frac = call <2 x float> @llvm.[[TARGET]].frac.v2f32
// CHECK: ret <2 x float> %hlsl.frac
float2 test_frac_float2(float2 p0) { return frac(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.frac = call <3 x float> @llvm.[[TARGET]].frac.v3f32
// CHECK: ret <3 x float> %hlsl.frac
float3 test_frac_float3(float3 p0) { return frac(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.frac = call <4 x float> @llvm.[[TARGET]].frac.v4f32
// CHECK: ret <4 x float> %hlsl.frac
float4 test_frac_float4(float4 p0) { return frac(p0); }
