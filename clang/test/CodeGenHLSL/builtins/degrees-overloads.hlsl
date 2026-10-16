// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s  \
// RUN:   -verify -verify-ignore-unexpected=note

// Note: the f0x42652EE1 constants below equal 180/Pi.

// CHECK-LABEL: test_degrees_double
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, f0x42652EE1
// CHECK:    ret float [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x 64 bit API lowering for degrees is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
float test_degrees_double(double p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_double2
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <2 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x 64 bit API lowering for degrees is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
float2 test_degrees_double2(double2 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_double3
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <3 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x 64 bit API lowering for degrees is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
float3 test_degrees_double3(double3 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_double4
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <4 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x 64 bit API lowering for degrees is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
float4 test_degrees_double4(double4 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_int
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} i32 %{{.*}} to float
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, f0x42652EE1
// CHECK:    ret float [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float test_degrees_int(int p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_int2
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <2 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float2 test_degrees_int2(int2 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_int3
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <3 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float3 test_degrees_int3(int3 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_int4
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <4 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float4 test_degrees_int4(int4 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_uint
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} i32 %{{.*}} to float
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, f0x42652EE1
// CHECK:    ret float [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float test_degrees_uint(uint p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_uint2
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <2 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float2 test_degrees_uint2(uint2 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_uint3
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <3 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float3 test_degrees_uint3(uint3 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_uint4
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <4 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float4 test_degrees_uint4(uint4 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_int64_t
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} i64 %{{.*}} to float
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, f0x42652EE1
// CHECK:    ret float [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float test_degrees_int64_t(int64_t p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_int64_t2
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <2 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float2 test_degrees_int64_t2(int64_t2 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_int64_t3
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <3 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float3 test_degrees_int64_t3(int64_t3 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_int64_t4
// CHECK:    [[CONVI:%.*]] = sitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <4 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float4 test_degrees_int64_t4(int64_t4 p0) { return degrees(p0); }

// CHECK-LABEL: test_degrees_uint64_t
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} i64 %{{.*}} to float
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, f0x42652EE1
// CHECK:    ret float [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float test_degrees_uint64_t(uint64_t p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_uint64_t2
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <2 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float2 test_degrees_uint64_t2(uint64_t2 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_uint64_t3
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <3 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float3 test_degrees_uint64_t3(uint64_t3 p0) { return degrees(p0); }
// CHECK-LABEL: test_degrees_uint64_t4
// CHECK:    [[CONVI:%.*]] = uitofp {{.*}} <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, splat (float f0x42652EE1)
// CHECK:    ret <4 x float> [[MUL]]
// expected-warning@+1 {{'degrees' is deprecated: In 202x int lowering for degrees is deprecated. Explicitly cast parameters to float types.}}
float4 test_degrees_uint64_t4(uint64_t4 p0) { return degrees(p0); }
