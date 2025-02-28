// Spirv target:
//
// ---------- Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -o - | FileCheck %s \
// RUN:   -DFNATTRS="spir_func noundef nofpclass(nan inf)" -DTYPE=half

//
// ---------- No Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm \
// RUN:   -o - | FileCheck %s \
// RUN:   -DFNATTRS="spir_func noundef nofpclass(nan inf)" -DTYPE=float



// CHECK: define [[FNATTRS]] [[TYPE]] @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn [[TYPE]]
// CHECK: ret [[TYPE]] %fmod.i
half test_fmod_half(half p0, half p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <2 x [[TYPE]]> @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]>
// CHECK: ret <2 x [[TYPE]]> %fmod.i
half2 test_fmod_half2(half2 p0, half2 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <3 x [[TYPE]]> @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]>
// CHECK: ret <3 x [[TYPE]]> %fmod.i
half3 test_fmod_half3(half3 p0, half3 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <4 x [[TYPE]]> @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]>
// CHECK: ret <4 x [[TYPE]]> %fmod.i
half4 test_fmod_half4(half4 p0, half4 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn float
// CHECK: ret float %fmod.i
float test_fmod_float(float p0, float p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <2 x float>
// CHECK: ret <2 x float> %fmod.i
float2 test_fmod_float2(float2 p0, float2 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <3 x float>
// CHECK: ret <3 x float> %fmod.i
float3 test_fmod_float3(float3 p0, float3 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <4 x float>
// CHECK: ret <4 x float> %fmod.i
float4 test_fmod_float4(float4 p0, float4 p1) { return fmod(p0, p1); }

