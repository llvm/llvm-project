// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:   -DFNATTRS="hidden noundef nofpclass(nan inf)" -DTARGET=dx
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:   -DFNATTRS="hidden spir_func noundef nofpclass(nan inf)" -DTARGET=spv
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s  \
// RUN:   -verify -verify-ignore-unexpected=note
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple spirv-unknown-vulkan-compute %s  \
// RUN:   -verify -verify-ignore-unexpected=note

// CHECK: define [[FNATTRS]] float @_Z16test_step_doubledd(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} float @llvm.[[TARGET]].step.f32(float [[CONVI]], float [[CONV1I]])
// CHECK:    ret float [[HLSLSTEPI]]
float test_step_double(double p0, double p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x 64 bit API lowering for step is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <2 x float> @_Z17test_step_double2Dv2_dS_(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].step.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[HLSLSTEPI]]
float2 test_step_double2(double2 p0, double2 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x 64 bit API lowering for step is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <3 x float> @_Z17test_step_double3Dv3_dS_(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].step.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[HLSLSTEPI]]
float3 test_step_double3(double3 p0, double3 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x 64 bit API lowering for step is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <4 x float> @_Z17test_step_double4Dv4_dS_(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].step.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[HLSLSTEPI]]
float4 test_step_double4(double4 p0, double4 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x 64 bit API lowering for step is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
    return step(p0, p1);
}

// CHECK: define [[FNATTRS]] float @_Z13test_step_intii(
// CHECK:    [[CONVI:%.*]] = sitofp i32 %{{.*}} to float
// CHECK:    [[CONV1I:%.*]] = sitofp i32 %{{.*}} to float
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} float @llvm.[[TARGET]].step.f32(float [[CONVI]], float [[CONV1I]])
// CHECK:    ret float [[HLSLSTEPI]]
float test_step_int(int p0, int p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <2 x float> @_Z14test_step_int2Dv2_iS_(
// CHECK:    [[CONVI:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].step.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[HLSLSTEPI]]
float2 test_step_int2(int2 p0, int2 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <3 x float> @_Z14test_step_int3Dv3_iS_(
// CHECK:    [[CONVI:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].step.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[HLSLSTEPI]]
float3 test_step_int3(int3 p0, int3 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <4 x float> @_Z14test_step_int4Dv4_iS_(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].step.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[HLSLSTEPI]]
float4 test_step_int4(int4 p0, int4 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}

// CHECK: define [[FNATTRS]] float @_Z14test_step_uintjj(
// CHECK:    [[CONVI:%.*]] = uitofp i32 %{{.*}} to float
// CHECK:    [[CONV1I:%.*]] = uitofp i32 %{{.*}} to float
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} float @llvm.[[TARGET]].step.f32(float [[CONVI]], float [[CONV1I]])
// CHECK:    ret float [[HLSLSTEPI]]
float test_step_uint(uint p0, uint p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <2 x float> @_Z15test_step_uint2Dv2_jS_(
// CHECK:    [[CONVI:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].step.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[HLSLSTEPI]]
float2 test_step_uint2(uint2 p0, uint2 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <3 x float> @_Z15test_step_uint3Dv3_jS_(
// CHECK:    [[CONVI:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].step.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[HLSLSTEPI]]
float3 test_step_uint3(uint3 p0, uint3 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <4 x float> @_Z15test_step_uint4Dv4_jS_(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].step.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[HLSLSTEPI]]
float4 test_step_uint4(uint4 p0, uint4 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}

// CHECK: define [[FNATTRS]] float @_Z17test_step_int64_tll(
// CHECK:    [[CONVI:%.*]] = sitofp i64 %{{.*}} to float
// CHECK:    [[CONV1I:%.*]] = sitofp i64 %{{.*}} to float
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} float @llvm.[[TARGET]].step.f32(float [[CONVI]], float [[CONV1I]])
// CHECK:    ret float [[HLSLSTEPI]]
float test_step_int64_t(int64_t p0, int64_t p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <2 x float> @_Z18test_step_int64_t2Dv2_lS_(
// CHECK:    [[CONVI:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].step.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[HLSLSTEPI]]
float2 test_step_int64_t2(int64_t2 p0, int64_t2 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <3 x float> @_Z18test_step_int64_t3Dv3_lS_(
// CHECK:    [[CONVI:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].step.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[HLSLSTEPI]]
float3 test_step_int64_t3(int64_t3 p0, int64_t3 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <4 x float> @_Z18test_step_int64_t4Dv4_lS_(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].step.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[HLSLSTEPI]]
float4 test_step_int64_t4(int64_t4 p0, int64_t4 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}

// CHECK: define [[FNATTRS]] float @_Z18test_step_uint64_tmm(
// CHECK:    [[CONVI:%.*]] = uitofp i64 %{{.*}} to float
// CHECK:    [[CONV1I:%.*]] = uitofp i64 %{{.*}} to float
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} float @llvm.[[TARGET]].step.f32(float [[CONVI]], float [[CONV1I]])
// CHECK:    ret float [[HLSLSTEPI]]
float test_step_uint64_t(uint64_t p0, uint64_t p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <2 x float> @_Z19test_step_uint64_t2Dv2_mS_(
// CHECK:    [[CONVI:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].step.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[HLSLSTEPI]]
float2 test_step_uint64_t2(uint64_t2 p0, uint64_t2 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <3 x float> @_Z19test_step_uint64_t3Dv3_mS_(
// CHECK:    [[CONVI:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].step.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[HLSLSTEPI]]
float3 test_step_uint64_t3(uint64_t3 p0, uint64_t3 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK: define [[FNATTRS]] <4 x float> @_Z19test_step_uint64_t4Dv4_mS_(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[HLSLSTEPI:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].step.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[HLSLSTEPI]]
float4 test_step_uint64_t4(uint64_t4 p0, uint64_t4 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
