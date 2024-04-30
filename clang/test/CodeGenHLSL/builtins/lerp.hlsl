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


// DXIL_NATIVE_HALF: %hlsl.lerp = call half @llvm.dx.lerp.f16(half %0, half %1, half %2)
// SPIR_NATIVE_HALF: %hlsl.lerp = call half @llvm.spv.lerp.f16(half %0, half %1, half %2)
// NATIVE_HALF: ret half %hlsl.lerp
// DXIL_NO_HALF: %hlsl.lerp = call float @llvm.dx.lerp.f32(float %0, float %1, float %2)
// SPIR_NO_HALF: %hlsl.lerp = call float @llvm.spv.lerp.f32(float %0, float %1, float %2)
// NO_HALF: ret float %hlsl.lerp
half test_lerp_half(half p0) { return lerp(p0, p0, p0); }

// DXIL_NATIVE_HALF: %hlsl.lerp = call <2 x half> @llvm.dx.lerp.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %2)
// SPIR_NATIVE_HALF: %hlsl.lerp = call <2 x half> @llvm.spv.lerp.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %2)
// NATIVE_HALF: ret <2 x half> %hlsl.lerp
// DXIL_NO_HALF: %hlsl.lerp = call <2 x float> @llvm.dx.lerp.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2)
// SPIR_NO_HALF: %hlsl.lerp = call <2 x float> @llvm.spv.lerp.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2)
// NO_HALF: ret <2 x float> %hlsl.lerp
half2 test_lerp_half2(half2 p0) { return lerp(p0, p0, p0); }

// DXIL_NATIVE_HALF: %hlsl.lerp = call <3 x half> @llvm.dx.lerp.v3f16(<3 x half> %0, <3 x half> %1, <3 x half> %2)
// SPIR_NATIVE_HALF: %hlsl.lerp = call <3 x half> @llvm.spv.lerp.v3f16(<3 x half> %0, <3 x half> %1, <3 x half> %2)
// NATIVE_HALF: ret <3 x half> %hlsl.lerp
// DXIL_NO_HALF: %hlsl.lerp = call <3 x float> @llvm.dx.lerp.v3f32(<3 x float> %0, <3 x float> %1, <3 x float> %2)
// SPIR_NO_HALF: %hlsl.lerp = call <3 x float> @llvm.spv.lerp.v3f32(<3 x float> %0, <3 x float> %1, <3 x float> %2)
// NO_HALF: ret <3 x float> %hlsl.lerp
half3 test_lerp_half3(half3 p0) { return lerp(p0, p0, p0); }

// DXIL_NATIVE_HALF: %hlsl.lerp = call <4 x half> @llvm.dx.lerp.v4f16(<4 x half> %0, <4 x half> %1, <4 x half> %2)
// SPIR_NATIVE_HALF: %hlsl.lerp = call <4 x half> @llvm.spv.lerp.v4f16(<4 x half> %0, <4 x half> %1, <4 x half> %2)
// NATIVE_HALF: ret <4 x half> %hlsl.lerp
// DXIL_NO_HALF: %hlsl.lerp = call <4 x float> @llvm.dx.lerp.v4f32(<4 x float> %0, <4 x float> %1, <4 x float> %2)
// SPIR_NO_HALF: %hlsl.lerp = call <4 x float> @llvm.spv.lerp.v4f32(<4 x float> %0, <4 x float> %1, <4 x float> %2)
// NO_HALF: ret <4 x float> %hlsl.lerp
half4 test_lerp_half4(half4 p0) { return lerp(p0, p0, p0); }

// DXIL_CHECK: %hlsl.lerp = call float @llvm.dx.lerp.f32(float %0, float %1, float %2)
// SPIR_CHECK: %hlsl.lerp = call float @llvm.spv.lerp.f32(float %0, float %1, float %2)
// CHECK: ret float %hlsl.lerp
float test_lerp_float(float p0) { return lerp(p0, p0, p0); }

// DXIL_CHECK: %hlsl.lerp = call <2 x float> @llvm.dx.lerp.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2)
// SPIR_CHECK: %hlsl.lerp = call <2 x float> @llvm.spv.lerp.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2)
// CHECK: ret <2 x float> %hlsl.lerp
float2 test_lerp_float2(float2 p0) { return lerp(p0, p0, p0); }

// DXIL_CHECK: %hlsl.lerp = call <3 x float> @llvm.dx.lerp.v3f32(<3 x float> %0, <3 x float> %1, <3 x float> %2)
// SPIR_CHECK: %hlsl.lerp = call <3 x float> @llvm.spv.lerp.v3f32(<3 x float> %0, <3 x float> %1, <3 x float> %2)
// CHECK: ret <3 x float> %hlsl.lerp
float3 test_lerp_float3(float3 p0) { return lerp(p0, p0, p0); }

// DXIL_CHECK: %hlsl.lerp = call <4 x float> @llvm.dx.lerp.v4f32(<4 x float> %0, <4 x float> %1, <4 x float> %2)
// SPIR_CHECK: %hlsl.lerp = call <4 x float> @llvm.spv.lerp.v4f32(<4 x float> %0, <4 x float> %1, <4 x float> %2)
// CHECK: ret <4 x float> %hlsl.lerp
float4 test_lerp_float4(float4 p0) { return lerp(p0, p0, p0); }

// DXIL_CHECK: %hlsl.lerp = call <2 x float> @llvm.dx.lerp.v2f32(<2 x float> %splat.splat, <2 x float> %1, <2 x float> %2)
// SPIR_CHECK: %hlsl.lerp = call <2 x float> @llvm.spv.lerp.v2f32(<2 x float> %splat.splat, <2 x float> %1, <2 x float> %2)
// CHECK: ret <2 x float> %hlsl.lerp
float2 test_lerp_float2_splat(float p0, float2 p1) { return lerp(p0, p1, p1); }

// DXIL_CHECK: %hlsl.lerp = call <3 x float> @llvm.dx.lerp.v3f32(<3 x float> %splat.splat, <3 x float> %1, <3 x float> %2)
// SPIR_CHECK: %hlsl.lerp = call <3 x float> @llvm.spv.lerp.v3f32(<3 x float> %splat.splat, <3 x float> %1, <3 x float> %2)
// CHECK: ret <3 x float> %hlsl.lerp
float3 test_lerp_float3_splat(float p0, float3 p1) { return lerp(p0, p1, p1); }

// DXIL_CHECK:  %hlsl.lerp = call <4 x float> @llvm.dx.lerp.v4f32(<4 x float> %splat.splat, <4 x float> %1, <4 x float> %2)
// SPIR_CHECK:  %hlsl.lerp = call <4 x float> @llvm.spv.lerp.v4f32(<4 x float> %splat.splat, <4 x float> %1, <4 x float> %2)
// CHECK:  ret <4 x float> %hlsl.lerp
float4 test_lerp_float4_splat(float p0, float4 p1) { return lerp(p0, p1, p1); }

// CHECK: %conv = sitofp i32 %2 to float
// CHECK: %splat.splatinsert = insertelement <2 x float> poison, float %conv, i64 0
// CHECK: %splat.splat = shufflevector <2 x float> %splat.splatinsert, <2 x float> poison, <2 x i32> zeroinitializer
// DXIL_CHECK: %hlsl.lerp = call <2 x float> @llvm.dx.lerp.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %splat.splat)
// SPIR_CHECK: %hlsl.lerp = call <2 x float> @llvm.spv.lerp.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %splat.splat)
// CHECK: ret <2 x float> %hlsl.lerp
float2 test_lerp_float2_int_splat(float2 p0, int p1) {
  return lerp(p0, p0, p1);
}

// CHECK: %conv = sitofp i32 %2 to float
// CHECK: %splat.splatinsert = insertelement <3 x float> poison, float %conv, i64 0
// CHECK: %splat.splat = shufflevector <3 x float> %splat.splatinsert, <3 x float> poison, <3 x i32> zeroinitializer
// DXIL_CHECK:  %hlsl.lerp = call <3 x float> @llvm.dx.lerp.v3f32(<3 x float> %0, <3 x float> %1, <3 x float> %splat.splat)
// SPIR_CHECK:  %hlsl.lerp = call <3 x float> @llvm.spv.lerp.v3f32(<3 x float> %0, <3 x float> %1, <3 x float> %splat.splat)
// CHECK: ret <3 x float> %hlsl.lerp
float3 test_lerp_float3_int_splat(float3 p0, int p1) {
  return lerp(p0, p0, p1);
}
