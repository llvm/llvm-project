// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,DXCHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,DXCHECK,NO_HALF

// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SPVCHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,SPVCHECK,NO_HALF

// DXCHECK: define hidden [[FN_TYPE:]]noundef i1 @
// SPVCHECK: define hidden [[FN_TYPE:spir_func ]]noundef i1 @
// DXCHECK: %hlsl.isinf = call i1 @llvm.[[ICF:dx]].isinf.f32(
// SPVCHECK: %hlsl.isinf = call i1 @llvm.[[ICF:spv]].isinf.f32(
// CHECK: ret i1 %hlsl.isinf
bool test_isinf_float(float p0) { return isinf(p0); }

// CHECK: define hidden [[FN_TYPE]]noundef i1 @
// NATIVE_HALF: %hlsl.isinf = call i1 @llvm.[[ICF]].isinf.f16(
// NO_HALF: %hlsl.isinf = call i1 @llvm.[[ICF]].isinf.f32(
// CHECK: ret i1 %hlsl.isinf
bool test_isinf_half(half p0) { return isinf(p0); }

// CHECK: define hidden [[FN_TYPE]]noundef <2 x i1> @
// NATIVE_HALF: %hlsl.isinf = call <2 x i1> @llvm.[[ICF]].isinf.v2f16
// NO_HALF: %hlsl.isinf = call <2 x i1> @llvm.[[ICF]].isinf.v2f32(
// CHECK: ret <2 x i1> %hlsl.isinf
bool2 test_isinf_half2(half2 p0) { return isinf(p0); }

// NATIVE_HALF: define hidden [[FN_TYPE]]noundef <3 x i1> @
// NATIVE_HALF: %hlsl.isinf = call <3 x i1> @llvm.[[ICF]].isinf.v3f16
// NO_HALF: %hlsl.isinf = call <3 x i1> @llvm.[[ICF]].isinf.v3f32(
// CHECK: ret <3 x i1> %hlsl.isinf
bool3 test_isinf_half3(half3 p0) { return isinf(p0); }

// NATIVE_HALF: define hidden [[FN_TYPE]]noundef <4 x i1> @
// NATIVE_HALF: %hlsl.isinf = call <4 x i1> @llvm.[[ICF]].isinf.v4f16
// NO_HALF: %hlsl.isinf = call <4 x i1> @llvm.[[ICF]].isinf.v4f32(
// CHECK: ret <4 x i1> %hlsl.isinf
bool4 test_isinf_half4(half4 p0) { return isinf(p0); }


// CHECK: define hidden [[FN_TYPE]]noundef <2 x i1> @
// CHECK: %hlsl.isinf = call <2 x i1> @llvm.[[ICF]].isinf.v2f32
// CHECK: ret <2 x i1> %hlsl.isinf
bool2 test_isinf_float2(float2 p0) { return isinf(p0); }

// CHECK: define hidden [[FN_TYPE]]noundef <3 x i1> @
// CHECK: %hlsl.isinf = call <3 x i1> @llvm.[[ICF]].isinf.v3f32
// CHECK: ret <3 x i1> %hlsl.isinf
bool3 test_isinf_float3(float3 p0) { return isinf(p0); }

// CHECK: define hidden [[FN_TYPE]]noundef <4 x i1> @
// CHECK: %hlsl.isinf = call <4 x i1> @llvm.[[ICF]].isinf.v4f32
// CHECK: ret <4 x i1> %hlsl.isinf
bool4 test_isinf_float4(float4 p0) { return isinf(p0); }
