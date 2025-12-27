// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:   -DFNATTRS="hidden noundef nofpclass(nan inf)" -DTARGET=dx
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:   -DFNATTRS="hidden spir_func noundef nofpclass(nan inf)" -DTARGET=spv

// CHECK: define [[FNATTRS]] float @
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].normalize.f32(float
// CHECK: ret float
float test_normalize_double(double p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].normalize.v2f32(<2 x float>
// CHECK: ret <2 x float> %hlsl.normalize
float2 test_normalize_double2(double2 p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].normalize.v3f32(
// CHECK: ret <3 x float> %hlsl.normalize
float3 test_normalize_double3(double3 p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].normalize.v4f32(
// CHECK: ret <4 x float> %hlsl.normalize
float4 test_length_double4(double4 p0)
{
    return normalize(p0);
}

// CHECK: define [[FNATTRS]] float @
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].normalize.f32(float
// CHECK: ret float
float test_normalize_int(int p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].normalize.v2f32(<2 x float>
// CHECK: ret <2 x float> %hlsl.normalize
float2 test_normalize_int2(int2 p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].normalize.v3f32(
// CHECK: ret <3 x float> %hlsl.normalize
float3 test_normalize_int3(int3 p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].normalize.v4f32(
// CHECK: ret <4 x float> %hlsl.normalize
float4 test_length_int4(int4 p0)
{
    return normalize(p0);
}

// CHECK: define [[FNATTRS]] float @
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].normalize.f32(float
// CHECK: ret float
float test_normalize_uint(uint p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].normalize.v2f32(<2 x float>

// CHECK: ret <2 x float> %hlsl.normalize
float2 test_normalize_uint2(uint2 p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].normalize.v3f32(
// CHECK: ret <3 x float> %hlsl.normalize
float3 test_normalize_uint3(uint3 p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].normalize.v4f32(
// CHECK: ret <4 x float> %hlsl.normalize
float4 test_length_uint4(uint4 p0)
{
    return normalize(p0);
}

// CHECK: define [[FNATTRS]] float @
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].normalize.f32(float
// CHECK: ret float
float test_normalize_int64_t(int64_t p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].normalize.v2f32(<2 x float>

// CHECK: ret <2 x float> %hlsl.normalize
float2 test_normalize_int64_t2(int64_t2 p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].normalize.v3f32(
// CHECK: ret <3 x float> %hlsl.normalize
float3 test_normalize_int64_t3(int64_t3 p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].normalize.v4f32(
// CHECK: ret <4 x float> %hlsl.normalize
float4 test_length_int64_t4(int64_t4 p0)
{
    return normalize(p0);
}

// CHECK: define [[FNATTRS]] float @
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].normalize.f32(float
// CHECK: ret float
float test_normalize_uint64_t(uint64_t p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].normalize.v2f32(<2 x float>

// CHECK: ret <2 x float> %hlsl.normalize
float2 test_normalize_uint64_t2(uint64_t2 p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].normalize.v3f32(
// CHECK: ret <3 x float> %hlsl.normalize
float3 test_normalize_uint64_t3(uint64_t3 p0)
{
    return normalize(p0);
}
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.normalize = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].normalize.v4f32(
// CHECK: ret <4 x float> %hlsl.normalize
float4 test_length_uint64_t4(uint64_t4 p0)
{
    return normalize(p0);
}
