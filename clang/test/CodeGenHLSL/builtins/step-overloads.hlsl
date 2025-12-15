// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:   -DFNATTRS="hidden noundef nofpclass(nan inf)" -DTARGET=dx
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:   -DFNATTRS="hidden spir_func noundef nofpclass(nan inf)" -DTARGET=spv

// CHECK: define [[FNATTRS]] float @
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].step.f32(float
// CHECK: ret float
float test_step_double(double p0, double p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].step.v2f32(
// CHECK: ret <2 x float> %hlsl.step
float2 test_step_double2(double2 p0, double2 p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].step.v3f32(
// CHECK: ret <3 x float> %hlsl.step
float3 test_step_double3(double3 p0, double3 p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].step.v4f32(
// CHECK: ret <4 x float> %hlsl.step
float4 test_step_double4(double4 p0, double4 p1)
{
    return step(p0, p1);
}

// CHECK: define [[FNATTRS]] float @
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].step.f32(float
// CHECK: ret float
float test_step_int(int p0, int p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].step.v2f32(
// CHECK: ret <2 x float> %hlsl.step
float2 test_step_int2(int2 p0, int2 p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].step.v3f32(
// CHECK: ret <3 x float> %hlsl.step
float3 test_step_int3(int3 p0, int3 p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].step.v4f32(
// CHECK: ret <4 x float> %hlsl.step
float4 test_step_int4(int4 p0, int4 p1)
{
    return step(p0, p1);
}

// CHECK: define [[FNATTRS]] float @
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].step.f32(float
// CHECK: ret float
float test_step_uint(uint p0, uint p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].step.v2f32(
// CHECK: ret <2 x float> %hlsl.step
float2 test_step_uint2(uint2 p0, uint2 p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].step.v3f32(
// CHECK: ret <3 x float> %hlsl.step
float3 test_step_uint3(uint3 p0, uint3 p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].step.v4f32(
// CHECK: ret <4 x float> %hlsl.step
float4 test_step_uint4(uint4 p0, uint4 p1)
{
    return step(p0, p1);
}

// CHECK: define [[FNATTRS]] float @
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].step.f32(float
// CHECK: ret float
float test_step_int64_t(int64_t p0, int64_t p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].step.v2f32(
// CHECK: ret <2 x float> %hlsl.step
float2 test_step_int64_t2(int64_t2 p0, int64_t2 p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].step.v3f32(
// CHECK: ret <3 x float> %hlsl.step
float3 test_step_int64_t3(int64_t3 p0, int64_t3 p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].step.v4f32(
// CHECK: ret <4 x float> %hlsl.step
float4 test_step_int64_t4(int64_t4 p0, int64_t4 p1)
{
    return step(p0, p1);
}

// CHECK: define [[FNATTRS]] float @
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].step.f32(float
// CHECK: ret float
float test_step_uint64_t(uint64_t p0, uint64_t p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].step.v2f32(
// CHECK: ret <2 x float> %hlsl.step
float2 test_step_uint64_t2(uint64_t2 p0, uint64_t2 p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].step.v3f32(
// CHECK: ret <3 x float> %hlsl.step
float3 test_step_uint64_t3(uint64_t3 p0, uint64_t3 p1)
{
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.step = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].step.v4f32(
// CHECK: ret <4 x float> %hlsl.step
float4 test_step_uint64_t4(uint64_t4 p0, uint64_t4 p1)
{
    return step(p0, p1);
}
