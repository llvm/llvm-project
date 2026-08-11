// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s  \
// RUN:   -verify -verify-ignore-unexpected=note

// CHECK-LABEL: test_step_double
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt float %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} i1 [[CMP]], float 0.000000e+00, float 1.000000e+00
// CHECK:    ret float [[SELECT]]
float test_step_double(double p0, double p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x 64 bit API lowering for step is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_double2
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <2 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <2 x i1> [[CMP]], <2 x float> zeroinitializer, <2 x float> splat (float 1.000000e+00)
// CHECK:    ret <2 x float> [[SELECT]]
float2 test_step_double2(double2 p0, double2 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x 64 bit API lowering for step is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_double3
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <3 x i1> [[CMP]], <3 x float> zeroinitializer, <3 x float> splat (float 1.000000e+00)
// CHECK:    ret <3 x float> [[SELECT]]
float3 test_step_double3(double3 p0, double3 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x 64 bit API lowering for step is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_double4
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <4 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <4 x i1> [[CMP]], <4 x float> zeroinitializer, <4 x float> splat (float 1.000000e+00)
// CHECK:    ret <4 x float> [[SELECT]]
float4 test_step_double4(double4 p0, double4 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x 64 bit API lowering for step is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
    return step(p0, p1);
}

// CHECK-LABEL: test_step_int
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} i32 %{{.*}} to float
// CHECK:    [[CONV1I:%.*]] = sitofp {{.*}} i32 %{{.*}} to float
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt float %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} i1 [[CMP]], float 0.000000e+00, float 1.000000e+00
// CHECK:    ret float [[SELECT]]
float test_step_int(int p0, int p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_int2
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <2 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <2 x i1> [[CMP]], <2 x float> zeroinitializer, <2 x float> splat (float 1.000000e+00)
// CHECK:    ret <2 x float> [[SELECT]]
float2 test_step_int2(int2 p0, int2 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_int3
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <3 x i1> [[CMP]], <3 x float> zeroinitializer, <3 x float> splat (float 1.000000e+00)
// CHECK:    ret <3 x float> [[SELECT]]
float3 test_step_int3(int3 p0, int3 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_int4
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <4 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <4 x i1> [[CMP]], <4 x float> zeroinitializer, <4 x float> splat (float 1.000000e+00)
// CHECK:    ret <4 x float> [[SELECT]]
float4 test_step_int4(int4 p0, int4 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}

// CHECK-LABEL: test_step_uint
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} i32 %{{.*}} to float
// CHECK:    [[CONV1I:%.*]] = uitofp {{.*}} i32 %{{.*}} to float
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt float %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} i1 [[CMP]], float 0.000000e+00, float 1.000000e+00
// CHECK:    ret float [[SELECT]]
float test_step_uint(uint p0, uint p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_uint2
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <2 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <2 x i1> [[CMP]], <2 x float> zeroinitializer, <2 x float> splat (float 1.000000e+00)
// CHECK:    ret <2 x float> [[SELECT]]
float2 test_step_uint2(uint2 p0, uint2 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_uint3
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <3 x i1> [[CMP]], <3 x float> zeroinitializer, <3 x float> splat (float 1.000000e+00)
// CHECK:    ret <3 x float> [[SELECT]]
float3 test_step_uint3(uint3 p0, uint3 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_uint4
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <4 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <4 x i1> [[CMP]], <4 x float> zeroinitializer, <4 x float> splat (float 1.000000e+00)
// CHECK:    ret <4 x float> [[SELECT]]
float4 test_step_uint4(uint4 p0, uint4 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}

// CHECK-LABEL: test_step_int64_t
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} i64 %{{.*}} to float
// CHECK:    [[CONV1I:%.*]] = sitofp {{.*}} i64 %{{.*}} to float
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt float %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} i1 [[CMP]], float 0.000000e+00, float 1.000000e+00
// CHECK:    ret float [[SELECT]]
float test_step_int64_t(int64_t p0, int64_t p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_int64_t2
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <2 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <2 x i1> [[CMP]], <2 x float> zeroinitializer, <2 x float> splat (float 1.000000e+00)
// CHECK:    ret <2 x float> [[SELECT]]
float2 test_step_int64_t2(int64_t2 p0, int64_t2 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_int64_t3
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <3 x i1> [[CMP]], <3 x float> zeroinitializer, <3 x float> splat (float 1.000000e+00)
// CHECK:    ret <3 x float> [[SELECT]]
float3 test_step_int64_t3(int64_t3 p0, int64_t3 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_int64_t4
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <4 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <4 x i1> [[CMP]], <4 x float> zeroinitializer, <4 x float> splat (float 1.000000e+00)
// CHECK:    ret <4 x float> [[SELECT]]
float4 test_step_int64_t4(int64_t4 p0, int64_t4 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}

// CHECK-LABEL: test_step_uint64_t
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} i64 %{{.*}} to float
// CHECK:    [[CONV1I:%.*]] = uitofp {{.*}} i64 %{{.*}} to float
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt float %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} i1 [[CMP]], float 0.000000e+00, float 1.000000e+00
// CHECK:    ret float [[SELECT]]
float test_step_uint64_t(uint64_t p0, uint64_t p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_uint64_t2
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <2 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <2 x i1> [[CMP]], <2 x float> zeroinitializer, <2 x float> splat (float 1.000000e+00)
// CHECK:    ret <2 x float> [[SELECT]]
float2 test_step_uint64_t2(uint64_t2 p0, uint64_t2 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_uint64_t3
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <3 x i1> [[CMP]], <3 x float> zeroinitializer, <3 x float> splat (float 1.000000e+00)
// CHECK:    ret <3 x float> [[SELECT]]
float3 test_step_uint64_t3(uint64_t3 p0, uint64_t3 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
// CHECK-LABEL: test_step_uint64_t4
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CMP:%.*]] = fcmp {{.*}} olt <4 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SELECT:%.*]] = select {{.*}} <4 x i1> [[CMP]], <4 x float> zeroinitializer, <4 x float> splat (float 1.000000e+00)
// CHECK:    ret <4 x float> [[SELECT]]
float4 test_step_uint64_t4(uint64_t4 p0, uint64_t4 p1)
{
// expected-warning@+1 {{'step' is deprecated: In 202x int lowering for step is deprecated. Explicitly cast parameters to float types.}}
    return step(p0, p1);
}
