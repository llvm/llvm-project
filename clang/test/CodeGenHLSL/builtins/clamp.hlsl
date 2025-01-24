// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF \
// RUN:  -DTARGET=dx -DFNATTRS=noundef -DFFNATTRS="nofpclass(nan inf)"
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:  -DTARGET=dx -DFNATTRS=noundef -DFFNATTRS="nofpclass(nan inf)"
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF \
// RUN:  -DTARGET=spv -DFNATTRS="spir_func noundef" -DFFNATTRS="nofpclass(nan inf)"
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:  -DTARGET=spv -DFNATTRS="spir_func noundef" -DFFNATTRS="nofpclass(nan inf)"

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF: define [[FNATTRS]] i16 @_Z16test_clamp_short
// NATIVE_HALF: call i16 @llvm.[[TARGET]].sclamp.i16(
int16_t test_clamp_short(int16_t p0, int16_t p1) { return clamp(p0, p1,p1); }
// NATIVE_HALF: define [[FNATTRS]] <2 x i16> @_Z17test_clamp_short2
// NATIVE_HALF: call <2 x i16> @llvm.[[TARGET]].sclamp.v2i16(
int16_t2 test_clamp_short2(int16_t2 p0, int16_t2 p1) { return clamp(p0, p1,p1); }
// NATIVE_HALF: define [[FNATTRS]] <3 x i16> @_Z17test_clamp_short3
// NATIVE_HALF: call <3 x i16> @llvm.[[TARGET]].sclamp.v3i16
int16_t3 test_clamp_short3(int16_t3 p0, int16_t3 p1) { return clamp(p0, p1,p1); }
// NATIVE_HALF: define [[FNATTRS]] <4 x i16> @_Z17test_clamp_short4
// NATIVE_HALF: call <4 x i16> @llvm.[[TARGET]].sclamp.v4i16
int16_t4 test_clamp_short4(int16_t4 p0, int16_t4 p1) { return clamp(p0, p1,p1); }

// NATIVE_HALF: define [[FNATTRS]] i16 @_Z17test_clamp_ushort
// NATIVE_HALF: call i16 @llvm.[[TARGET]].uclamp.i16(
uint16_t test_clamp_ushort(uint16_t p0, uint16_t p1) { return clamp(p0, p1,p1); }
// NATIVE_HALF: define [[FNATTRS]] <2 x i16> @_Z18test_clamp_ushort2
// NATIVE_HALF: call <2 x i16> @llvm.[[TARGET]].uclamp.v2i16
uint16_t2 test_clamp_ushort2(uint16_t2 p0, uint16_t2 p1) { return clamp(p0, p1,p1); }
// NATIVE_HALF: define [[FNATTRS]] <3 x i16> @_Z18test_clamp_ushort3
// NATIVE_HALF: call <3 x i16> @llvm.[[TARGET]].uclamp.v3i16
uint16_t3 test_clamp_ushort3(uint16_t3 p0, uint16_t3 p1) { return clamp(p0, p1,p1); }
// NATIVE_HALF: define [[FNATTRS]] <4 x i16> @_Z18test_clamp_ushort4
// NATIVE_HALF: call <4 x i16> @llvm.[[TARGET]].uclamp.v4i16
uint16_t4 test_clamp_ushort4(uint16_t4 p0, uint16_t4 p1) { return clamp(p0, p1,p1); }
#endif

// CHECK: define [[FNATTRS]] i32 @_Z14test_clamp_int
// CHECK: call i32 @llvm.[[TARGET]].sclamp.i32(
int test_clamp_int(int p0, int p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] <2 x i32> @_Z15test_clamp_int2
// CHECK: call <2 x i32> @llvm.[[TARGET]].sclamp.v2i32
int2 test_clamp_int2(int2 p0, int2 p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] <3 x i32> @_Z15test_clamp_int3
// CHECK: call <3 x i32> @llvm.[[TARGET]].sclamp.v3i32
int3 test_clamp_int3(int3 p0, int3 p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] <4 x i32> @_Z15test_clamp_int4
// CHECK: call <4 x i32> @llvm.[[TARGET]].sclamp.v4i32
int4 test_clamp_int4(int4 p0, int4 p1) { return clamp(p0, p1,p1); }

// CHECK: define [[FNATTRS]] i32 @_Z15test_clamp_uint
// CHECK: call i32 @llvm.[[TARGET]].uclamp.i32(
int test_clamp_uint(uint p0, uint p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] <2 x i32> @_Z16test_clamp_uint2
// CHECK: call <2 x i32> @llvm.[[TARGET]].uclamp.v2i32
uint2 test_clamp_uint2(uint2 p0, uint2 p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] <3 x i32> @_Z16test_clamp_uint3
// CHECK: call <3 x i32> @llvm.[[TARGET]].uclamp.v3i32
uint3 test_clamp_uint3(uint3 p0, uint3 p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] <4 x i32> @_Z16test_clamp_uint4
// CHECK: call <4 x i32> @llvm.[[TARGET]].uclamp.v4i32
uint4 test_clamp_uint4(uint4 p0, uint4 p1) { return clamp(p0, p1,p1); }

// CHECK: define [[FNATTRS]] i64 @_Z15test_clamp_long
// CHECK: call i64 @llvm.[[TARGET]].sclamp.i64(
int64_t test_clamp_long(int64_t p0, int64_t p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] <2 x i64> @_Z16test_clamp_long2
// CHECK: call <2 x i64> @llvm.[[TARGET]].sclamp.v2i64
int64_t2 test_clamp_long2(int64_t2 p0, int64_t2 p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] <3 x i64> @_Z16test_clamp_long3
// CHECK: call <3 x i64> @llvm.[[TARGET]].sclamp.v3i64
int64_t3 test_clamp_long3(int64_t3 p0, int64_t3 p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] <4 x i64> @_Z16test_clamp_long4
// CHECK: call <4 x i64> @llvm.[[TARGET]].sclamp.v4i64
int64_t4 test_clamp_long4(int64_t4 p0, int64_t4 p1) { return clamp(p0, p1,p1); }

// CHECK: define [[FNATTRS]] i64 @_Z16test_clamp_ulong
// CHECK: call i64 @llvm.[[TARGET]].uclamp.i64(
uint64_t test_clamp_ulong(uint64_t p0, uint64_t p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] <2 x i64> @_Z17test_clamp_ulong2
// CHECK: call <2 x i64> @llvm.[[TARGET]].uclamp.v2i64
uint64_t2 test_clamp_ulong2(uint64_t2 p0, uint64_t2 p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] <3 x i64> @_Z17test_clamp_ulong3
// CHECK: call <3 x i64> @llvm.[[TARGET]].uclamp.v3i64
uint64_t3 test_clamp_ulong3(uint64_t3 p0, uint64_t3 p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] <4 x i64> @_Z17test_clamp_ulong4
// CHECK: call <4 x i64> @llvm.[[TARGET]].uclamp.v4i64
uint64_t4 test_clamp_ulong4(uint64_t4 p0, uint64_t4 p1) { return clamp(p0, p1,p1); }

// NATIVE_HALF: define [[FNATTRS]] [[FFNATTRS]] half @_Z15test_clamp_half
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn half @llvm.[[TARGET]].nclamp.f16(
// NO_HALF: define [[FNATTRS]] [[FFNATTRS]] float @_Z15test_clamp_half
// NO_HALF: call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].nclamp.f32(
half test_clamp_half(half p0, half p1) { return clamp(p0, p1,p1); }
// NATIVE_HALF: define [[FNATTRS]] [[FFNATTRS]] <2 x half> @_Z16test_clamp_half2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.[[TARGET]].nclamp.v2f16
// NO_HALF: define [[FNATTRS]] [[FFNATTRS]] <2 x float> @_Z16test_clamp_half2
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].nclamp.v2f32(
half2 test_clamp_half2(half2 p0, half2 p1) { return clamp(p0, p1,p1); }
// NATIVE_HALF: define [[FNATTRS]] [[FFNATTRS]] <3 x half> @_Z16test_clamp_half3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.[[TARGET]].nclamp.v3f16
// NO_HALF: define [[FNATTRS]] [[FFNATTRS]] <3 x float> @_Z16test_clamp_half3
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].nclamp.v3f32(
half3 test_clamp_half3(half3 p0, half3 p1) { return clamp(p0, p1,p1); }
// NATIVE_HALF: define [[FNATTRS]] [[FFNATTRS]] <4 x half> @_Z16test_clamp_half4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.[[TARGET]].nclamp.v4f16
// NO_HALF: define [[FNATTRS]] [[FFNATTRS]] <4 x float> @_Z16test_clamp_half4
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].nclamp.v4f32(
half4 test_clamp_half4(half4 p0, half4 p1) { return clamp(p0, p1,p1); }

// CHECK: define [[FNATTRS]] [[FFNATTRS]] float @_Z16test_clamp_float
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].nclamp.f32(
float test_clamp_float(float p0, float p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] [[FFNATTRS]] <2 x float> @_Z17test_clamp_float2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].nclamp.v2f32
float2 test_clamp_float2(float2 p0, float2 p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] [[FFNATTRS]] <3 x float> @_Z17test_clamp_float3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].nclamp.v3f32
float3 test_clamp_float3(float3 p0, float3 p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] [[FFNATTRS]] <4 x float> @_Z17test_clamp_float4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].nclamp.v4f32
float4 test_clamp_float4(float4 p0, float4 p1) { return clamp(p0, p1,p1); }

// CHECK: define [[FNATTRS]] [[FFNATTRS]] double @_Z17test_clamp_double
// CHECK: call reassoc nnan ninf nsz arcp afn double @llvm.[[TARGET]].nclamp.f64(
double test_clamp_double(double p0, double p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] [[FFNATTRS]] <2 x double> @_Z18test_clamp_double2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.[[TARGET]].nclamp.v2f64
double2 test_clamp_double2(double2 p0, double2 p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] [[FFNATTRS]] <3 x double> @_Z18test_clamp_double3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.[[TARGET]].nclamp.v3f64
double3 test_clamp_double3(double3 p0, double3 p1) { return clamp(p0, p1,p1); }
// CHECK: define [[FNATTRS]] [[FFNATTRS]] <4 x double> @_Z18test_clamp_double4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.[[TARGET]].nclamp.v4f64
double4 test_clamp_double4(double4 p0, double4 p1) { return clamp(p0, p1,p1); }
