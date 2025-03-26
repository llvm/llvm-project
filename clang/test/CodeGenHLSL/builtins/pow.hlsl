// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) half @_Z13test_pow_half
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn half @llvm.pow.f16(
// NO_HALF-LABEL: define noundef nofpclass(nan inf) float @_Z13test_pow_half
// NO_HALF: call reassoc nnan ninf nsz arcp afn float @llvm.pow.f32(
half test_pow_half(half p0, half p1) { return pow(p0, p1); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <2 x half> @_Z14test_pow_half2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.pow.v2f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z14test_pow_half2
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.pow.v2f32(
half2 test_pow_half2(half2 p0, half2 p1) { return pow(p0, p1); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <3 x half> @_Z14test_pow_half3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.pow.v3f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z14test_pow_half3
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.pow.v3f32(
half3 test_pow_half3(half3 p0, half3 p1) { return pow(p0, p1); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <4 x half> @_Z14test_pow_half4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.pow.v4f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z14test_pow_half4
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.pow.v4f32(
half4 test_pow_half4(half4 p0, half4 p1) { return pow(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float @_Z14test_pow_float
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.pow.f32(
float test_pow_float(float p0, float p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z15test_pow_float2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.pow.v2f32
float2 test_pow_float2(float2 p0, float2 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z15test_pow_float3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.pow.v3f32
float3 test_pow_float3(float3 p0, float3 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z15test_pow_float4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.pow.v4f32
float4 test_pow_float4(float4 p0, float4 p1) { return pow(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_pow_double
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.pow.f32(
float test_pow_double(double p0, double p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_pow_double2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.pow.v2f32
float2 test_pow_double2(double2 p0, double2 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_pow_double3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.pow.v3f32
float3 test_pow_double3(double3 p0, double3 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_pow_double4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.pow.v4f32
float4 test_pow_double4(double4 p0, double4 p1) { return pow(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_pow_int
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.pow.f32(
float test_pow_int(int p0, int p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_pow_int2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.pow.v2f32
float2 test_pow_int2(int2 p0, int2 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_pow_int3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.pow.v3f32
float3 test_pow_int3(int3 p0, int3 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_pow_int4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.pow.v4f32
float4 test_pow_int4(int4 p0, int4 p1) { return pow(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_pow_uint
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.pow.f32(
float test_pow_uint(uint p0, uint p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_pow_uint2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.pow.v2f32
float2 test_pow_uint2(uint2 p0, uint2 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_pow_uint3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.pow.v3f32
float3 test_pow_uint3(uint3 p0, uint3 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_pow_uint4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.pow.v4f32
float4 test_pow_uint4(uint4 p0, uint4 p1) { return pow(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_pow_int64_t
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.pow.f32(
float test_pow_int64_t(int64_t p0, int64_t p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_pow_int64_t2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.pow.v2f32
float2 test_pow_int64_t2(int64_t2 p0, int64_t2 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_pow_int64_t3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.pow.v3f32
float3 test_pow_int64_t3(int64_t3 p0, int64_t3 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_pow_int64_t4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.pow.v4f32
float4 test_pow_int64_t4(int64_t4 p0, int64_t4 p1) { return pow(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_pow_uint64_t
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.pow.f32(
float test_pow_uint64_t(uint64_t p0, uint64_t p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_pow_uint64_t2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.pow.v2f32
float2 test_pow_uint64_t2(uint64_t2 p0, uint64_t2 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_pow_uint64_t3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.pow.v3f32
float3 test_pow_uint64_t3(uint64_t3 p0, uint64_t3 p1) { return pow(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_pow_uint64_t4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.pow.v4f32
float4 test_pow_uint64_t4(uint64_t4 p0, uint64_t4 p1) { return pow(p0, p1); }
