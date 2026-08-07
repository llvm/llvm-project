// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,DXCHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,DXCHECK,NO_HALF

// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SPVCHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,SPVCHECK,NO_HALF

// CHECK-LABEL: define hidden {{.*}}noundef <2 x i1> @_{{.*}}test_isnan_float1x2{{.*}}(
// DXCHECK: %hlsl.isnan = call <2 x i1> @llvm.[[ICF:dx]].isnan.v2f32(
// SPVCHECK: %hlsl.isnan = call <2 x i1> @llvm.[[ICF:spv]].isnan.v2f32(
// CHECK: ret <2 x i1> %hlsl.isnan
bool1x2 test_isnan_float1x2(float1x2 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <3 x i1> @_{{.*}}test_isnan_float1x3{{.*}}(
// CHECK: %hlsl.isnan = call <3 x i1> @llvm.[[ICF]].isnan.v3f32
// CHECK: ret <3 x i1> %hlsl.isnan
bool1x3 test_isnan_float1x3(float1x3 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <4 x i1> @_{{.*}}test_isnan_float1x4{{.*}}(
// CHECK: %hlsl.isnan = call <4 x i1> @llvm.[[ICF]].isnan.v4f32
// CHECK: ret <4 x i1> %hlsl.isnan
bool1x4 test_isnan_float1x4(float1x4 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <2 x i1> @_{{.*}}test_isnan_float2x1{{.*}}(
// CHECK: %hlsl.isnan = call <2 x i1> @llvm.[[ICF]].isnan.v2f32
// CHECK: ret <2 x i1> %hlsl.isnan
bool2x1 test_isnan_float2x1(float2x1 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <4 x i1> @_{{.*}}test_isnan_float2x2{{.*}}(
// CHECK: %hlsl.isnan = call <4 x i1> @llvm.[[ICF]].isnan.v4f32
// CHECK: ret <4 x i1> %hlsl.isnan
bool2x2 test_isnan_float2x2(float2x2 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <6 x i1> @_{{.*}}test_isnan_float2x3{{.*}}(
// CHECK: %hlsl.isnan = call <6 x i1> @llvm.[[ICF]].isnan.v6f32
// CHECK: ret <6 x i1> %hlsl.isnan
bool2x3 test_isnan_float2x3(float2x3 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <8 x i1> @_{{.*}}test_isnan_float2x4{{.*}}(
// CHECK: %hlsl.isnan = call <8 x i1> @llvm.[[ICF]].isnan.v8f32
// CHECK: ret <8 x i1> %hlsl.isnan
bool2x4 test_isnan_float2x4(float2x4 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <3 x i1> @_{{.*}}test_isnan_float3x1{{.*}}(
// CHECK: %hlsl.isnan = call <3 x i1> @llvm.[[ICF]].isnan.v3f32
// CHECK: ret <3 x i1> %hlsl.isnan
bool3x1 test_isnan_float3x1(float3x1 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <6 x i1> @_{{.*}}test_isnan_float3x2{{.*}}(
// CHECK: %hlsl.isnan = call <6 x i1> @llvm.[[ICF]].isnan.v6f32
// CHECK: ret <6 x i1> %hlsl.isnan
bool3x2 test_isnan_float3x2(float3x2 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <9 x i1> @_{{.*}}test_isnan_float3x3{{.*}}(
// CHECK: %hlsl.isnan = call <9 x i1> @llvm.[[ICF]].isnan.v9f32
// CHECK: ret <9 x i1> %hlsl.isnan
bool3x3 test_isnan_float3x3(float3x3 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <12 x i1> @_{{.*}}test_isnan_float3x4{{.*}}(
// CHECK: %hlsl.isnan = call <12 x i1> @llvm.[[ICF]].isnan.v12f32
// CHECK: ret <12 x i1> %hlsl.isnan
bool3x4 test_isnan_float3x4(float3x4 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <4 x i1> @_{{.*}}test_isnan_float4x1{{.*}}(
// CHECK: %hlsl.isnan = call <4 x i1> @llvm.[[ICF]].isnan.v4f32
// CHECK: ret <4 x i1> %hlsl.isnan
bool4x1 test_isnan_float4x1(float4x1 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <8 x i1> @_{{.*}}test_isnan_float4x2{{.*}}(
// CHECK: %hlsl.isnan = call <8 x i1> @llvm.[[ICF]].isnan.v8f32
// CHECK: ret <8 x i1> %hlsl.isnan
bool4x2 test_isnan_float4x2(float4x2 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <12 x i1> @_{{.*}}test_isnan_float4x3{{.*}}(
// CHECK: %hlsl.isnan = call <12 x i1> @llvm.[[ICF]].isnan.v12f32
// CHECK: ret <12 x i1> %hlsl.isnan
bool4x3 test_isnan_float4x3(float4x3 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <16 x i1> @_{{.*}}test_isnan_float4x4{{.*}}(
// CHECK: %hlsl.isnan = call <16 x i1> @llvm.[[ICF]].isnan.v16f32
// CHECK: ret <16 x i1> %hlsl.isnan
bool4x4 test_isnan_float4x4(float4x4 p0) { return isnan(p0); }


// CHECK-LABEL: define hidden {{.*}}noundef <2 x i1> @_{{.*}}test_isnan_half1x2{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <2 x i1> @llvm.[[ICF]].isnan.v2f16
// NO_HALF: %hlsl.isnan = call <2 x i1> @llvm.[[ICF]].isnan.v2f32
// CHECK: ret <2 x i1> %hlsl.isnan
bool1x2 test_isnan_half1x2(half1x2 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <3 x i1> @_{{.*}}test_isnan_half1x3{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <3 x i1> @llvm.[[ICF]].isnan.v3f16
// NO_HALF: %hlsl.isnan = call <3 x i1> @llvm.[[ICF]].isnan.v3f32
// CHECK: ret <3 x i1> %hlsl.isnan
bool1x3 test_isnan_half1x3(half1x3 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <4 x i1> @_{{.*}}test_isnan_half1x4{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <4 x i1> @llvm.[[ICF]].isnan.v4f16
// NO_HALF: %hlsl.isnan = call <4 x i1> @llvm.[[ICF]].isnan.v4f32
// CHECK: ret <4 x i1> %hlsl.isnan
bool1x4 test_isnan_half1x4(half1x4 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <2 x i1> @_{{.*}}test_isnan_half2x1{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <2 x i1> @llvm.[[ICF]].isnan.v2f16
// NO_HALF: %hlsl.isnan = call <2 x i1> @llvm.[[ICF]].isnan.v2f32
// CHECK: ret <2 x i1> %hlsl.isnan
bool2x1 test_isnan_half2x1(half2x1 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <4 x i1> @_{{.*}}test_isnan_half2x2{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <4 x i1> @llvm.[[ICF]].isnan.v4f16
// NO_HALF: %hlsl.isnan = call <4 x i1> @llvm.[[ICF]].isnan.v4f32
// CHECK: ret <4 x i1> %hlsl.isnan
bool2x2 test_isnan_half2x2(half2x2 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <6 x i1> @_{{.*}}test_isnan_half2x3{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <6 x i1> @llvm.[[ICF]].isnan.v6f16
// NO_HALF: %hlsl.isnan = call <6 x i1> @llvm.[[ICF]].isnan.v6f32
// CHECK: ret <6 x i1> %hlsl.isnan
bool2x3 test_isnan_half2x3(half2x3 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <8 x i1> @_{{.*}}test_isnan_half2x4{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <8 x i1> @llvm.[[ICF]].isnan.v8f16
// NO_HALF: %hlsl.isnan = call <8 x i1> @llvm.[[ICF]].isnan.v8f32
// CHECK: ret <8 x i1> %hlsl.isnan
bool2x4 test_isnan_half2x4(half2x4 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <3 x i1> @_{{.*}}test_isnan_half3x1{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <3 x i1> @llvm.[[ICF]].isnan.v3f16
// NO_HALF: %hlsl.isnan = call <3 x i1> @llvm.[[ICF]].isnan.v3f32
// CHECK: ret <3 x i1> %hlsl.isnan
bool3x1 test_isnan_half3x1(half3x1 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <6 x i1> @_{{.*}}test_isnan_half3x2{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <6 x i1> @llvm.[[ICF]].isnan.v6f16
// NO_HALF: %hlsl.isnan = call <6 x i1> @llvm.[[ICF]].isnan.v6f32
// CHECK: ret <6 x i1> %hlsl.isnan
bool3x2 test_isnan_half3x2(half3x2 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <9 x i1> @_{{.*}}test_isnan_half3x3{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <9 x i1> @llvm.[[ICF]].isnan.v9f16
// NO_HALF: %hlsl.isnan = call <9 x i1> @llvm.[[ICF]].isnan.v9f32
// CHECK: ret <9 x i1> %hlsl.isnan
bool3x3 test_isnan_half3x3(half3x3 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <12 x i1> @_{{.*}}test_isnan_half3x4{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <12 x i1> @llvm.[[ICF]].isnan.v12f16
// NO_HALF: %hlsl.isnan = call <12 x i1> @llvm.[[ICF]].isnan.v12f32
// CHECK: ret <12 x i1> %hlsl.isnan
bool3x4 test_isnan_half3x4(half3x4 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <4 x i1> @_{{.*}}test_isnan_half4x1{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <4 x i1> @llvm.[[ICF]].isnan.v4f16
// NO_HALF: %hlsl.isnan = call <4 x i1> @llvm.[[ICF]].isnan.v4f32
// CHECK: ret <4 x i1> %hlsl.isnan
bool4x1 test_isnan_half4x1(half4x1 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <8 x i1> @_{{.*}}test_isnan_half4x2{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <8 x i1> @llvm.[[ICF]].isnan.v8f16
// NO_HALF: %hlsl.isnan = call <8 x i1> @llvm.[[ICF]].isnan.v8f32
// CHECK: ret <8 x i1> %hlsl.isnan
bool4x2 test_isnan_half4x2(half4x2 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <12 x i1> @_{{.*}}test_isnan_half4x3{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <12 x i1> @llvm.[[ICF]].isnan.v12f16
// NO_HALF: %hlsl.isnan = call <12 x i1> @llvm.[[ICF]].isnan.v12f32
// CHECK: ret <12 x i1> %hlsl.isnan
bool4x3 test_isnan_half4x3(half4x3 p0) { return isnan(p0); }

// CHECK-LABEL: define hidden {{.*}}noundef <16 x i1> @_{{.*}}test_isnan_half4x4{{.*}}(
// NATIVE_HALF: %hlsl.isnan = call <16 x i1> @llvm.[[ICF]].isnan.v16f16
// NO_HALF: %hlsl.isnan = call <16 x i1> @llvm.[[ICF]].isnan.v16f32
// CHECK: ret <16 x i1> %hlsl.isnan
bool4x4 test_isnan_half4x4(half4x4 p0) { return isnan(p0); }
