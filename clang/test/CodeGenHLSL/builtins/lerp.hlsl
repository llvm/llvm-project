// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF: %3 = fsub half %1, %0
// NATIVE_HALF: %4 = fmul half %2, %3
// NATIVE_HALF: %dx.lerp = fadd half %0, %4
// NATIVE_HALF: ret half %dx.lerp
// NO_HALF: %3 = fsub float %1, %0
// NO_HALF: %4 = fmul float %2, %3
// NO_HALF: %dx.lerp = fadd float %0, %4
// NO_HALF: ret float %dx.lerp
half test_lerp_half(half p0) { return lerp(p0, p0, p0); }

// NATIVE_HALF: %dx.lerp = call <2 x half> @llvm.dx.lerp.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %2)
// NATIVE_HALF: ret <2 x half> %dx.lerp
// NO_HALF: %dx.lerp = call <2 x float> @llvm.dx.lerp.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2)
// NO_HALF: ret <2 x float> %dx.lerp
half2 test_lerp_half2(half2 p0, half2 p1) { return lerp(p0, p0, p0); }

// NATIVE_HALF: %dx.lerp = call <3 x half> @llvm.dx.lerp.v3f16(<3 x half> %0, <3 x half> %1, <3 x half> %2)
// NATIVE_HALF: ret <3 x half> %dx.lerp
// NO_HALF: %dx.lerp = call <3 x float> @llvm.dx.lerp.v3f32(<3 x float> %0, <3 x float> %1, <3 x float> %2)
// NO_HALF: ret <3 x float> %dx.lerp
half3 test_lerp_half3(half3 p0, half3 p1) { return lerp(p0, p0, p0); }

// NATIVE_HALF: %dx.lerp = call <4 x half> @llvm.dx.lerp.v4f16(<4 x half> %0, <4 x half> %1, <4 x half> %2)
// NATIVE_HALF: ret <4 x half> %dx.lerp
// NO_HALF: %dx.lerp = call <4 x float> @llvm.dx.lerp.v4f32(<4 x float> %0, <4 x float> %1, <4 x float> %2)
// NO_HALF: ret <4 x float> %dx.lerp
half4 test_lerp_half4(half4 p0, half4 p1) { return lerp(p0, p0, p0); }

// CHECK: %3 = fsub float %1, %0
// CHECK: %4 = fmul float %2, %3
// CHECK: %dx.lerp = fadd float %0, %4
// CHECK: ret float %dx.lerp
float test_lerp_float(float p0, float p1) { return lerp(p0, p0, p0); }

// CHECK: %dx.lerp = call <2 x float> @llvm.dx.lerp.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2)
// CHECK: ret <2 x float> %dx.lerp
float2 test_lerp_float2(float2 p0, float2 p1) { return lerp(p0, p0, p0); }

// CHECK: %dx.lerp = call <3 x float> @llvm.dx.lerp.v3f32(<3 x float> %0, <3 x float> %1, <3 x float> %2)
// CHECK: ret <3 x float> %dx.lerp
float3 test_lerp_float3(float3 p0, float3 p1) { return lerp(p0, p0, p0); }

// CHECK: %dx.lerp = call <4 x float> @llvm.dx.lerp.v4f32(<4 x float> %0, <4 x float> %1, <4 x float> %2)
// CHECK: ret <4 x float> %dx.lerp
float4 test_lerp_float4(float4 p0, float4 p1) { return lerp(p0, p0, p0); }

// CHECK: %dx.lerp = call <2 x float> @llvm.dx.lerp.v2f32(<2 x float> %splat.splat, <2 x float> %1, <2 x float> %2)
// CHECK: ret <2 x float> %dx.lerp
float2 test_lerp_float2_splat(float p0, float2 p1) { return lerp(p0, p1, p1); }

// CHECK: %dx.lerp = call <3 x float> @llvm.dx.lerp.v3f32(<3 x float> %splat.splat, <3 x float> %1, <3 x float> %2)
// CHECK: ret <3 x float> %dx.lerp
float3 test_lerp_float3_splat(float p0, float3 p1) { return lerp(p0, p1, p1); }

// CHECK:  %dx.lerp = call <4 x float> @llvm.dx.lerp.v4f32(<4 x float> %splat.splat, <4 x float> %1, <4 x float> %2)
// CHECK:  ret <4 x float> %dx.lerp
float4 test_lerp_float4_splat(float p0, float4 p1) { return lerp(p0, p1, p1); }

// CHECK: %conv = sitofp i32 %2 to float
// CHECK: %splat.splatinsert = insertelement <2 x float> poison, float %conv, i64 0
// CHECK: %splat.splat = shufflevector <2 x float> %splat.splatinsert, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK: %dx.lerp = call <2 x float> @llvm.dx.lerp.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %splat.splat)
// CHECK: ret <2 x float> %dx.lerp
float2 test_lerp_float2_int_splat(float2 p0, int p1) {
  return lerp(p0, p0, p1);
}

// CHECK: %conv = sitofp i32 %2 to float
// CHECK: %splat.splatinsert = insertelement <3 x float> poison, float %conv, i64 0
// CHECK: %splat.splat = shufflevector <3 x float> %splat.splatinsert, <3 x float> poison, <3 x i32> zeroinitializer
// CHECK:  %dx.lerp = call <3 x float> @llvm.dx.lerp.v3f32(<3 x float> %0, <3 x float> %1, <3 x float> %splat.splat)
// CHECK: ret <3 x float> %dx.lerp
float3 test_lerp_float3_int_splat(float3 p0, int p1) {
  return lerp(p0, p0, p1);
}
