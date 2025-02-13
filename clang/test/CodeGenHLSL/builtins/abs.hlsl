// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF

using hlsl::abs;

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF-LABEL: define noundef i16 @_Z16test_abs_int16_t
// NATIVE_HALF: call i16 @llvm.abs.i16(
int16_t test_abs_int16_t(int16_t p0) { return abs(p0); }
// NATIVE_HALF-LABEL: define noundef <2 x i16> @_Z17test_abs_int16_t2
// NATIVE_HALF: call <2 x i16> @llvm.abs.v2i16(
int16_t2 test_abs_int16_t2(int16_t2 p0) { return abs(p0); }
// NATIVE_HALF-LABEL: define noundef <3 x i16> @_Z17test_abs_int16_t3
// NATIVE_HALF: call <3 x i16> @llvm.abs.v3i16(
int16_t3 test_abs_int16_t3(int16_t3 p0) { return abs(p0); }
// NATIVE_HALF-LABEL: define noundef <4 x i16> @_Z17test_abs_int16_t4
// NATIVE_HALF: call <4 x i16> @llvm.abs.v4i16(
int16_t4 test_abs_int16_t4(int16_t4 p0) { return abs(p0); }
#endif // __HLSL_ENABLE_16_BIT

// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) half @_Z13test_abs_half
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn half @llvm.fabs.f16(
// NO_HALF-LABEL: define noundef nofpclass(nan inf) float @_Z13test_abs_half
// NO_HALF: call reassoc nnan ninf nsz arcp afn float @llvm.fabs.f32(float %0)
half test_abs_half(half p0) { return abs(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <2 x half> @_Z14test_abs_half2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.fabs.v2f16(
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z14test_abs_half2
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.fabs.v2f32(
half2 test_abs_half2(half2 p0) { return abs(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <3 x half> @_Z14test_abs_half3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.fabs.v3f16(
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z14test_abs_half3
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.fabs.v3f32(
half3 test_abs_half3(half3 p0) { return abs(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <4 x half> @_Z14test_abs_half4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.fabs.v4f16(
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z14test_abs_half4
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.fabs.v4f32(
half4 test_abs_half4(half4 p0) { return abs(p0); }

// CHECK-LABEL: define noundef i32 @_Z12test_abs_int
// CHECK: call i32 @llvm.abs.i32(
int test_abs_int(int p0) { return abs(p0); }
// CHECK-LABEL: define noundef <2 x i32> @_Z13test_abs_int2
// CHECK: call <2 x i32> @llvm.abs.v2i32(
int2 test_abs_int2(int2 p0) { return abs(p0); }
// CHECK-LABEL: define noundef <3 x i32> @_Z13test_abs_int3
// CHECK: call <3 x i32> @llvm.abs.v3i32(
int3 test_abs_int3(int3 p0) { return abs(p0); }
// CHECK-LABEL: define noundef <4 x i32> @_Z13test_abs_int4
// CHECK: call <4 x i32> @llvm.abs.v4i32(
int4 test_abs_int4(int4 p0) { return abs(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float @_Z14test_abs_float
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.fabs.f32(
float test_abs_float(float p0) { return abs(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z15test_abs_float2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.fabs.v2f32(
float2 test_abs_float2(float2 p0) { return abs(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z15test_abs_float3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.fabs.v3f32(
float3 test_abs_float3(float3 p0) { return abs(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z15test_abs_float4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.fabs.v4f32(
float4 test_abs_float4(float4 p0) { return abs(p0); }

// CHECK-LABEL: define noundef i64 @_Z16test_abs_int64_t
// CHECK: call i64 @llvm.abs.i64(
int64_t test_abs_int64_t(int64_t p0) { return abs(p0); }
// CHECK-LABEL: define noundef <2 x i64> @_Z17test_abs_int64_t2
// CHECK: call <2 x i64> @llvm.abs.v2i64(
int64_t2 test_abs_int64_t2(int64_t2 p0) { return abs(p0); }
// CHECK-LABEL: define noundef <3 x i64> @_Z17test_abs_int64_t3
// CHECK: call <3 x i64> @llvm.abs.v3i64(
int64_t3 test_abs_int64_t3(int64_t3 p0) { return abs(p0); }
// CHECK-LABEL: define noundef <4 x i64> @_Z17test_abs_int64_t4
// CHECK: call <4 x i64> @llvm.abs.v4i64(
int64_t4 test_abs_int64_t4(int64_t4 p0) { return abs(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) double @_Z15test_abs_double
// CHECK: call reassoc nnan ninf nsz arcp afn double @llvm.fabs.f64(
double test_abs_double(double p0) { return abs(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x double> @_Z16test_abs_double2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.fabs.v2f64(
double2 test_abs_double2(double2 p0) { return abs(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x double> @_Z16test_abs_double3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.fabs.v3f64(
double3 test_abs_double3(double3 p0) { return abs(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x double> @_Z16test_abs_double4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.fabs.v4f64(
double4 test_abs_double4(double4 p0) { return abs(p0); }
