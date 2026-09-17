// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s --check-prefixes=CHECK,DXCHECK
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s --check-prefixes=CHECK,SPVCHECK
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s  \
// RUN:   -verify -verify-ignore-unexpected=note
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple spirv-unknown-vulkan-library %s  \
// RUN:   -verify -verify-ignore-unexpected=note

// CHECK-LABEL: test_normalize_double
// CHECK: [[CONVI:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK: [[ABS:%.*]] = call {{.*}} float @llvm.fabs.f32(float %{{.*}})
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} float %{{.*}}, [[ABS]]
// CHECK-NEXT: ret float [[RET]]
float test_normalize_double(double p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x 64 bit API lowering for normalize is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_double2
// CHECK: [[CONVI:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v2f32(<2 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <2 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <2 x float> [[SPLATINSERT]], <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <2 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <2 x float> [[RET]]
float2 test_normalize_double2(double2 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x 64 bit API lowering for normalize is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_double3
// CHECK: [[CONVI:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v3f32(<3 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <3 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <3 x float> [[SPLATINSERT]], <3 x float> poison, <3 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <3 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <3 x float> [[RET]]
float3 test_normalize_double3(double3 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x 64 bit API lowering for normalize is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_double4
// CHECK: [[CONVI:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v4f32(<4 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <4 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <4 x float> [[SPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <4 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <4 x float> [[RET]]
float4 test_normalize_double4(double4 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x 64 bit API lowering for normalize is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_int
// CHECK: [[CONVI:%.*]] = sitofp {{.*}} i32 %{{.*}} to float
// CHECK: [[ABS:%.*]] = call {{.*}} float @llvm.fabs.f32(float %{{.*}})
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} float %{{.*}}, [[ABS]]
// CHECK-NEXT: ret float [[RET]]
float test_normalize_int(int p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_int2
// CHECK: [[CONVI:%.*]] = sitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v2f32(<2 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <2 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <2 x float> [[SPLATINSERT]], <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <2 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <2 x float> [[RET]]
float2 test_normalize_int2(int2 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_int3
// CHECK: [[CONVI:%.*]] = sitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v3f32(<3 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <3 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <3 x float> [[SPLATINSERT]], <3 x float> poison, <3 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <3 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <3 x float> [[RET]]
float3 test_normalize_int3(int3 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_int4
// CHECK: [[CONVI:%.*]] = sitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v4f32(<4 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <4 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <4 x float> [[SPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <4 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <4 x float> [[RET]]
float4 test_normalize_int4(int4 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_uint
// CHECK: [[CONVI:%.*]] = uitofp {{.*}} i32 %{{.*}} to float
// CHECK: [[ABS:%.*]] = call {{.*}} float @llvm.fabs.f32(float %{{.*}})
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} float %{{.*}}, [[ABS]]
// CHECK-NEXT: ret float [[RET]]
float test_normalize_uint(uint p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_uint2
// CHECK: [[CONVI:%.*]] = uitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v2f32(<2 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <2 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <2 x float> [[SPLATINSERT]], <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <2 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <2 x float> [[RET]]
float2 test_normalize_uint2(uint2 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_uint3
// CHECK: [[CONVI:%.*]] = uitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v3f32(<3 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <3 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <3 x float> [[SPLATINSERT]], <3 x float> poison, <3 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <3 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <3 x float> [[RET]]
float3 test_normalize_uint3(uint3 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_uint4
// CHECK: [[CONVI:%.*]] = uitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v4f32(<4 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <4 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <4 x float> [[SPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <4 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <4 x float> [[RET]]
float4 test_normalize_uint4(uint4 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_int64_t
// CHECK: [[CONVI:%.*]] = sitofp {{.*}} i64 %{{.*}} to float
// CHECK: [[ABS:%.*]] = call {{.*}} float @llvm.fabs.f32(float %{{.*}})
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} float %{{.*}}, [[ABS]]
// CHECK-NEXT: ret float [[RET]]
float test_normalize_int64_t(int64_t p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_int64_t2
// CHECK: [[CONVI:%.*]] = sitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v2f32(<2 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <2 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <2 x float> [[SPLATINSERT]], <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <2 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <2 x float> [[RET]]
float2 test_normalize_int64_t2(int64_t2 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_int64_t3
// CHECK: [[CONVI:%.*]] = sitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v3f32(<3 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <3 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <3 x float> [[SPLATINSERT]], <3 x float> poison, <3 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <3 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <3 x float> [[RET]]
float3 test_normalize_int64_t3(int64_t3 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_int64_t4
// CHECK: [[CONVI:%.*]] = sitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v4f32(<4 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <4 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <4 x float> [[SPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <4 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <4 x float> [[RET]]
float4 test_normalize_int64_t4(int64_t4 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_uint64_t
// CHECK: [[CONVI:%.*]] = uitofp {{.*}} i64 %{{.*}} to float
// CHECK: [[ABS:%.*]] = call {{.*}} float @llvm.fabs.f32(float %{{.*}})
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} float %{{.*}}, [[ABS]]
// CHECK-NEXT: ret float [[RET]]
float test_normalize_uint64_t(uint64_t p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_uint64_t2
// CHECK: [[CONVI:%.*]] = uitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v2f32(<2 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <2 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <2 x float> [[SPLATINSERT]], <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <2 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <2 x float> [[RET]]
float2 test_normalize_uint64_t2(uint64_t2 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_uint64_t3
// CHECK: [[CONVI:%.*]] = uitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v3f32(<3 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <3 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <3 x float> [[SPLATINSERT]], <3 x float> poison, <3 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <3 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <3 x float> [[RET]]
float3 test_normalize_uint64_t3(uint64_t3 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}

// CHECK-LABEL: test_normalize_uint64_t4
// CHECK: [[CONVI:%.*]] = uitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// DXCHECK: [[DOT:%.*]] = call {{.*}} float @llvm.dx.fdot.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// DXCHECK-NEXT: [[LEN:%.*]] = call {{.*}} float @llvm.sqrt.f32(float [[DOT]])
// SPVCHECK: [[LEN:%.*]] = call {{.*}} float @llvm.spv.length.v4f32(<4 x float> %{{.*}})
// CHECK-NEXT: [[SPLATINSERT:%.*]] = insertelement <4 x float> poison, float [[LEN]], i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <4 x float> [[SPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK-NEXT: [[RET:%.*]] = fdiv {{.*}} <4 x float> %{{.*}}, [[SPLAT]]
// CHECK-NEXT: ret <4 x float> [[RET]]
float4 test_normalize_uint64_t4(uint64_t4 p0)
{
// expected-warning@+1 {{'normalize' is deprecated: In 202x int lowering for normalize is deprecated. Explicitly cast parameters to float types.}}
    return normalize(p0);
}
