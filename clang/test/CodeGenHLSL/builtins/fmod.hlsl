// DirectX target:
//
// ---------- Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -DFNATTRS="noundef nofpclass(nan inf)" -DTYPE=half

//
// ---------- No Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s \
// RUN:   -DFNATTRS="noundef nofpclass(nan inf)" -DTYPE=float


// Spirv target:
//
// ---------- Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -DFNATTRS="spir_func noundef nofpclass(nan inf)" -DTYPE=half

//
// ---------- No Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s \
// RUN:   -DFNATTRS="spir_func noundef nofpclass(nan inf)" -DTYPE=float



// CHECK: define [[FNATTRS]] [[TYPE]] @
// CHECK: call reassoc nnan ninf nsz arcp afn [[FNATTRS]] [[TYPE]] @{{.*}}([[TYPE]] noundef nofpclass(nan inf) %{{.*}}, [[TYPE]] noundef nofpclass(nan inf) %{{.*}}) #{{.*}}
// CHECK: ret [[TYPE]] %call
half test_fmod_half(half p0, half p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <2 x [[TYPE]]> @
// CHECK: call reassoc nnan ninf nsz arcp afn [[FNATTRS]] [[TYPE]] @{{.*}}(<2 x [[TYPE]]> noundef nofpclass(nan inf) %{{.*}}, <2 x [[TYPE]]> noundef nofpclass(nan inf) %{{.*}}) #{{.*}}
// CHECK: ret <2 x [[TYPE]]> %splat.splat
half2 test_fmod_half2(half2 p0, half2 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <3 x [[TYPE]]> @
// CHECK: call reassoc nnan ninf nsz arcp afn [[FNATTRS]] [[TYPE]] @{{.*}}(<3 x [[TYPE]]> noundef nofpclass(nan inf) %{{.*}}, <3 x [[TYPE]]> noundef nofpclass(nan inf) %{{.*}} #{{.*}}
// CHECK: ret <3 x [[TYPE]]> %splat.splat
half3 test_fmod_half3(half3 p0, half3 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <4 x [[TYPE]]> @
// CHECK: call reassoc nnan ninf nsz arcp afn [[FNATTRS]] [[TYPE]] @{{.*}}(<4 x [[TYPE]]> noundef nofpclass(nan inf) %{{.*}}, <4 x [[TYPE]]> noundef nofpclass(nan inf) %{{.*}}) #{{.*}}
// CHECK: ret <4 x [[TYPE]]> %splat.splat
half4 test_fmod_half4(half4 p0, half4 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] float @
// CHECK: call reassoc nnan ninf nsz arcp afn [[FNATTRS]] float @{{.*}}(float noundef nofpclass(nan inf) %{{.*}}, float noundef nofpclass(nan inf) %{{.*}}) #{{.*}}
// CHECK: ret float %call
float test_fmod_float(float p0, float p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: call reassoc nnan ninf nsz arcp afn [[FNATTRS]] float @{{.*}}(<2 x float> noundef nofpclass(nan inf) %{{.*}}, <2 x float> noundef nofpclass(nan inf) %{{.*}}) #{{.*}}
// CHECK: ret <2 x float> %splat.splat
float2 test_fmod_float2(float2 p0, float2 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: call reassoc nnan ninf nsz arcp afn [[FNATTRS]] float @{{.*}}(<3 x float> noundef nofpclass(nan inf) %{{.*}}, <3 x float> noundef nofpclass(nan inf) %{{.*}}) #{{.*}}
// CHECK: ret <3 x float> %splat.splat
float3 test_fmod_float3(float3 p0, float3 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: call reassoc nnan ninf nsz arcp afn [[FNATTRS]] float @{{.*}}(<4 x float> noundef nofpclass(nan inf) %{{.*}}, <4 x float> noundef nofpclass(nan inf) %{{.*}}) #{{.*}}
// CHECK: ret <4 x float> %splat.splat
float4 test_fmod_float4(float4 p0, float4 p1) { return fmod(p0, p1); }

