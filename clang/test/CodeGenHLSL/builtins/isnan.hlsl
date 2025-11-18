// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,DXCHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,DXCHECK,NO_HALF

// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SPVCHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,SPVCHECK,NO_HALF

// DXCHECK: define hidden [[FN_TYPE:]]noundef i1 @
// SPVCHECK: define hidden [[FN_TYPE:spir_func ]]noundef i1 @
// DXCHECK: %hlsl.isnan = call i1 @llvm.[[ICF:dx]].isnan.f32(
// SPVCHECK: %hlsl.isnan = call i1 @llvm.[[ICF:spv]].isnan.f32(
// CHECK: ret i1 %hlsl.isnan
bool test_isnan_float(float p0) { return isnan(p0); }

// CHECK: define hidden [[FN_TYPE]]noundef i1 @
// NATIVE_HALF: %hlsl.isnan = call i1 @llvm.[[ICF]].isnan.f16(
// NO_HALF: %hlsl.isnan = call i1 @llvm.[[ICF]].isnan.f32(
// CHECK: ret i1 %hlsl.isnan
bool test_isnan_half(half p0) { return isnan(p0); }

// CHECK: define hidden [[FN_TYPE]]noundef <2 x i1> @
// NATIVE_HALF: %hlsl.isnan = call <2 x i1> @llvm.[[ICF]].isnan.v2f16
// NO_HALF: %hlsl.isnan = call <2 x i1> @llvm.[[ICF]].isnan.v2f32(
// CHECK: ret <2 x i1> %hlsl.isnan
bool2 test_isnan_half2(half2 p0) { return isnan(p0); }

// NATIVE_HALF: define hidden [[FN_TYPE]]noundef <3 x i1> @
// NATIVE_HALF: %hlsl.isnan = call <3 x i1> @llvm.[[ICF]].isnan.v3f16
// NO_HALF: %hlsl.isnan = call <3 x i1> @llvm.[[ICF]].isnan.v3f32(
// CHECK: ret <3 x i1> %hlsl.isnan
bool3 test_isnan_half3(half3 p0) { return isnan(p0); }

// NATIVE_HALF: define hidden [[FN_TYPE]]noundef <4 x i1> @
// NATIVE_HALF: %hlsl.isnan = call <4 x i1> @llvm.[[ICF]].isnan.v4f16
// NO_HALF: %hlsl.isnan = call <4 x i1> @llvm.[[ICF]].isnan.v4f32(
// CHECK: ret <4 x i1> %hlsl.isnan
bool4 test_isnan_half4(half4 p0) { return isnan(p0); }


// CHECK: define hidden [[FN_TYPE]]noundef <2 x i1> @
// CHECK: %hlsl.isnan = call <2 x i1> @llvm.[[ICF]].isnan.v2f32
// CHECK: ret <2 x i1> %hlsl.isnan
bool2 test_isnan_float2(float2 p0) { return isnan(p0); }

// CHECK: define hidden [[FN_TYPE]]noundef <3 x i1> @
// CHECK: %hlsl.isnan = call <3 x i1> @llvm.[[ICF]].isnan.v3f32
// CHECK: ret <3 x i1> %hlsl.isnan
bool3 test_isnan_float3(float3 p0) { return isnan(p0); }

// CHECK: define hidden [[FN_TYPE]]noundef <4 x i1> @
// CHECK: %hlsl.isnan = call <4 x i1> @llvm.[[ICF]].isnan.v4f32
// CHECK: ret <4 x i1> %hlsl.isnan
bool4 test_isnan_float4(float4 p0) { return isnan(p0); }
