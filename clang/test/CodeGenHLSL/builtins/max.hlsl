// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF-LABEL: define noundef i16 @_Z14test_max_short
// NATIVE_HALF: call i16 @llvm.smax.i16(
int16_t test_max_short(int16_t p0, int16_t p1) { return max(p0, p1); }
// NATIVE_HALF-LABEL: define noundef <2 x i16> @_Z15test_max_short2
// NATIVE_HALF: call <2 x i16> @llvm.smax.v2i16(
int16_t2 test_max_short2(int16_t2 p0, int16_t2 p1) { return max(p0, p1); }
// NATIVE_HALF-LABEL: define noundef <3 x i16> @_Z15test_max_short3
// NATIVE_HALF: call <3 x i16> @llvm.smax.v3i16
int16_t3 test_max_short3(int16_t3 p0, int16_t3 p1) { return max(p0, p1); }
// NATIVE_HALF-LABEL: define noundef <4 x i16> @_Z15test_max_short4
// NATIVE_HALF: call <4 x i16> @llvm.smax.v4i16
int16_t4 test_max_short4(int16_t4 p0, int16_t4 p1) { return max(p0, p1); }

// NATIVE_HALF-LABEL: define noundef i16 @_Z15test_max_ushort
// NATIVE_HALF: call i16 @llvm.umax.i16(
uint16_t test_max_ushort(uint16_t p0, uint16_t p1) { return max(p0, p1); }
// NATIVE_HALF-LABEL: define noundef <2 x i16> @_Z16test_max_ushort2
// NATIVE_HALF: call <2 x i16> @llvm.umax.v2i16
uint16_t2 test_max_ushort2(uint16_t2 p0, uint16_t2 p1) { return max(p0, p1); }
// NATIVE_HALF-LABEL: define noundef <3 x i16> @_Z16test_max_ushort3
// NATIVE_HALF: call <3 x i16> @llvm.umax.v3i16
uint16_t3 test_max_ushort3(uint16_t3 p0, uint16_t3 p1) { return max(p0, p1); }
// NATIVE_HALF-LABEL: define noundef <4 x i16> @_Z16test_max_ushort4
// NATIVE_HALF: call <4 x i16> @llvm.umax.v4i16
uint16_t4 test_max_ushort4(uint16_t4 p0, uint16_t4 p1) { return max(p0, p1); }
#endif

// CHECK-LABEL: define noundef i32 @_Z12test_max_int
// CHECK: call i32 @llvm.smax.i32(
int test_max_int(int p0, int p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef <2 x i32> @_Z13test_max_int2
// CHECK: call <2 x i32> @llvm.smax.v2i32
int2 test_max_int2(int2 p0, int2 p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef <3 x i32> @_Z13test_max_int3
// CHECK: call <3 x i32> @llvm.smax.v3i32
int3 test_max_int3(int3 p0, int3 p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef <4 x i32> @_Z13test_max_int4
// CHECK: call <4 x i32> @llvm.smax.v4i32
int4 test_max_int4(int4 p0, int4 p1) { return max(p0, p1); }

// CHECK-LABEL: define noundef i32 @_Z13test_max_uint
// CHECK: call i32 @llvm.umax.i32(
int test_max_uint(uint p0, uint p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef <2 x i32> @_Z14test_max_uint2
// CHECK: call <2 x i32> @llvm.umax.v2i32
uint2 test_max_uint2(uint2 p0, uint2 p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef <3 x i32> @_Z14test_max_uint3
// CHECK: call <3 x i32> @llvm.umax.v3i32
uint3 test_max_uint3(uint3 p0, uint3 p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef <4 x i32> @_Z14test_max_uint4
// CHECK: call <4 x i32> @llvm.umax.v4i32
uint4 test_max_uint4(uint4 p0, uint4 p1) { return max(p0, p1); }

// CHECK-LABEL: define noundef i64 @_Z13test_max_long
// CHECK: call i64 @llvm.smax.i64(
int64_t test_max_long(int64_t p0, int64_t p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef <2 x i64> @_Z14test_max_long2
// CHECK: call <2 x i64> @llvm.smax.v2i64
int64_t2 test_max_long2(int64_t2 p0, int64_t2 p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef <3 x i64> @_Z14test_max_long3
// CHECK: call <3 x i64> @llvm.smax.v3i64
int64_t3 test_max_long3(int64_t3 p0, int64_t3 p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef <4 x i64> @_Z14test_max_long4
// CHECK: call <4 x i64> @llvm.smax.v4i64
int64_t4 test_max_long4(int64_t4 p0, int64_t4 p1) { return max(p0, p1); }

// CHECK-LABEL: define noundef i64 @_Z14test_max_ulong
// CHECK: call i64 @llvm.umax.i64(
uint64_t test_max_ulong(uint64_t p0, uint64_t p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef <2 x i64> @_Z15test_max_ulong2
// CHECK: call <2 x i64> @llvm.umax.v2i64
uint64_t2 test_max_ulong2(uint64_t2 p0, uint64_t2 p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef <3 x i64> @_Z15test_max_ulong3
// CHECK: call <3 x i64> @llvm.umax.v3i64
uint64_t3 test_max_ulong3(uint64_t3 p0, uint64_t3 p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef <4 x i64> @_Z15test_max_ulong4
// CHECK: call <4 x i64> @llvm.umax.v4i64
uint64_t4 test_max_ulong4(uint64_t4 p0, uint64_t4 p1) { return max(p0, p1); }

// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) half @_Z13test_max_half
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn half @llvm.maxnum.f16(
// NO_HALF-LABEL: define noundef nofpclass(nan inf) float @_Z13test_max_half
// NO_HALF: call reassoc nnan ninf nsz arcp afn float @llvm.maxnum.f32(
half test_max_half(half p0, half p1) { return max(p0, p1); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <2 x half> @_Z14test_max_half2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.maxnum.v2f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z14test_max_half2
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.maxnum.v2f32(
half2 test_max_half2(half2 p0, half2 p1) { return max(p0, p1); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <3 x half> @_Z14test_max_half3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.maxnum.v3f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z14test_max_half3
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.maxnum.v3f32(
half3 test_max_half3(half3 p0, half3 p1) { return max(p0, p1); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <4 x half> @_Z14test_max_half4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.maxnum.v4f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z14test_max_half4
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.maxnum.v4f32(
half4 test_max_half4(half4 p0, half4 p1) { return max(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float @_Z14test_max_float
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.maxnum.f32(
float test_max_float(float p0, float p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z15test_max_float2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.maxnum.v2f32
float2 test_max_float2(float2 p0, float2 p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z15test_max_float3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.maxnum.v3f32
float3 test_max_float3(float3 p0, float3 p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z15test_max_float4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.maxnum.v4f32
float4 test_max_float4(float4 p0, float4 p1) { return max(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) double @_Z15test_max_double
// CHECK: call reassoc nnan ninf nsz arcp afn double @llvm.maxnum.f64(
double test_max_double(double p0, double p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x double> @_Z16test_max_double2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.maxnum.v2f64
double2 test_max_double2(double2 p0, double2 p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x double> @_Z16test_max_double3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.maxnum.v3f64
double3 test_max_double3(double3 p0, double3 p1) { return max(p0, p1); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x double> @_Z16test_max_double4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.maxnum.v4f64
double4 test_max_double4(double4 p0, double4 p1) { return max(p0, p1); }
