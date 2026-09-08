// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s  \
// RUN:   -verify -verify-ignore-unexpected=note

// CHECK-LABEL: test_lerp_double
// CHECK:    [[CONV0:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[CONV1:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[CONV2:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn float %{{.*}}, [[MUL]]
// CHECK:    ret float [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x 64 bit API lowering for lerp is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
float test_lerp_double(double x, double y, double s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_double2
// CHECK:    [[CONV0:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[CONV1:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[CONV2:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <2 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x 64 bit API lowering for lerp is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
float2 test_lerp_double2(double2 x, double2 y, double2 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_double3
// CHECK:    [[CONV0:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[CONV1:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[CONV2:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <3 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x 64 bit API lowering for lerp is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
float3 test_lerp_double3(double3 x, double3 y, double3 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_double4
// CHECK:    [[CONV0:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[CONV1:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[CONV2:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <4 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x 64 bit API lowering for lerp is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
float4 test_lerp_double4(double4 x, double4 y, double4 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_int
// CHECK:    [[CONV0:%.*]] = sitofp {{.*}} i32 %{{.*}} to float
// CHECK:    [[CONV1:%.*]] = sitofp {{.*}} i32 %{{.*}} to float
// CHECK:    [[CONV2:%.*]] = sitofp {{.*}} i32 %{{.*}} to float
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn float %{{.*}}, [[MUL]]
// CHECK:    ret float [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float test_lerp_int(int x, int y, int s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_int2
// CHECK:    [[CONV0:%.*]] = sitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV1:%.*]] = sitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV2:%.*]] = sitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <2 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float2 test_lerp_int2(int2 x, int2 y, int2 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_int3
// CHECK:    [[CONV0:%.*]] = sitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV1:%.*]] = sitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV2:%.*]] = sitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <3 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float3 test_lerp_int3(int3 x, int3 y, int3 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_int4
// CHECK:    [[CONV0:%.*]] = sitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1:%.*]] = sitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV2:%.*]] = sitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <4 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float4 test_lerp_int4(int4 x, int4 y, int4 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_uint
// CHECK:    [[CONV0:%.*]] = uitofp {{.*}} i32 %{{.*}} to float
// CHECK:    [[CONV1:%.*]] = uitofp {{.*}} i32 %{{.*}} to float
// CHECK:    [[CONV2:%.*]] = uitofp {{.*}} i32 %{{.*}} to float
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn float %{{.*}}, [[MUL]]
// CHECK:    ret float [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float test_lerp_uint(uint x, uint y, uint s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_uint2
// CHECK:    [[CONV0:%.*]] = uitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV1:%.*]] = uitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV2:%.*]] = uitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <2 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float2 test_lerp_uint2(uint2 x, uint2 y, uint2 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_uint3
// CHECK:    [[CONV0:%.*]] = uitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV1:%.*]] = uitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV2:%.*]] = uitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <3 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float3 test_lerp_uint3(uint3 x, uint3 y, uint3 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_uint4
// CHECK:    [[CONV0:%.*]] = uitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1:%.*]] = uitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV2:%.*]] = uitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <4 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float4 test_lerp_uint4(uint4 x, uint4 y, uint4 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_int64_t
// CHECK:    [[CONV0:%.*]] = sitofp {{.*}} i64 %{{.*}} to float
// CHECK:    [[CONV1:%.*]] = sitofp {{.*}} i64 %{{.*}} to float
// CHECK:    [[CONV2:%.*]] = sitofp {{.*}} i64 %{{.*}} to float
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn float %{{.*}}, [[MUL]]
// CHECK:    ret float [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float test_lerp_int64_t(int64_t x, int64_t y, int64_t s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_int64_t2
// CHECK:    [[CONV0:%.*]] = sitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV1:%.*]] = sitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV2:%.*]] = sitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <2 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float2 test_lerp_int64_t2(int64_t2 x, int64_t2 y, int64_t2 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_int64_t3
// CHECK:    [[CONV0:%.*]] = sitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV1:%.*]] = sitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV2:%.*]] = sitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <3 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float3 test_lerp_int64_t3(int64_t3 x, int64_t3 y, int64_t3 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_int64_t4
// CHECK:    [[CONV0:%.*]] = sitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1:%.*]] = sitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV2:%.*]] = sitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <4 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float4 test_lerp_int64_t4(int64_t4 x, int64_t4 y, int64_t4 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_uint64_t
// CHECK:    [[CONV0:%.*]] = uitofp {{.*}} i64 %{{.*}} to float
// CHECK:    [[CONV1:%.*]] = uitofp {{.*}} i64 %{{.*}} to float
// CHECK:    [[CONV2:%.*]] = uitofp {{.*}} i64 %{{.*}} to float
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn float %{{.*}}, [[MUL]]
// CHECK:    ret float [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float test_lerp_uint64_t(uint64_t x, uint64_t y, uint64_t s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_uint64_t2
// CHECK:    [[CONV0:%.*]] = uitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV1:%.*]] = uitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV2:%.*]] = uitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <2 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float2 test_lerp_uint64_t2(uint64_t2 x, uint64_t2 y, uint64_t2 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_uint64_t3
// CHECK:    [[CONV0:%.*]] = uitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV1:%.*]] = uitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV2:%.*]] = uitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <3 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float3 test_lerp_uint64_t3(uint64_t3 x, uint64_t3 y, uint64_t3 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_uint64_t4
// CHECK:    [[CONV0:%.*]] = uitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1:%.*]] = uitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV2:%.*]] = uitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <4 x float> [[ADD]]
// expected-warning@+1 {{'lerp' is deprecated: In 202x int lowering for lerp is deprecated. Explicitly cast parameters to float types.}}
float4 test_lerp_uint64_t4(uint64_t4 x, uint64_t4 y, uint64_t4 s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_half_scalar
// CHECK:    [[SPLATINSERT:%.*]] = insertelement <3 x half> poison, half %{{.*}}, i64 0
// CHECK:    [[SPLAT:%.*]] = shufflevector <3 x half> [[SPLATINSERT]], <3 x half> poison, <3 x i32> zeroinitializer
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x half> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x half> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <3 x half> %{{.*}}, [[MUL]]
// CHECK:    ret <3 x half> [[ADD]]
// expected-warning@+1 {{'lerp<half, 3U>' is deprecated: In 202x mismatched vector/scalar lowering for lerp is deprecated. Explicitly cast parameters.}}
half3 test_lerp_half_scalar(half3 x, half3 y, half s) { return lerp(x, y, s); }

// CHECK-LABEL: test_lerp_float_scalar
// CHECK:    [[SPLATINSERT:%.*]] = insertelement <3 x float> poison, float %{{.*}}, i64 0
// CHECK:    [[SPLAT:%.*]] = shufflevector <3 x float> [[SPLATINSERT]], <3 x float> poison, <3 x i32> zeroinitializer
// CHECK:    [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <3 x float> [[ADD]]
// expected-warning@+1 {{'lerp<float, 3U>' is deprecated: In 202x mismatched vector/scalar lowering for lerp is deprecated. Explicitly cast parameters.}}
float3 test_lerp_float_scalar(float3 x, float3 y, float s) { return lerp(x, y, s); }
