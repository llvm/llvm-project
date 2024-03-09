// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF: %dx.umad = call i16 @llvm.dx.umad.i16(i16 %0, i16 %1, i16 %2)
// NATIVE_HALF: ret i16 %dx.umad
uint16_t test_mad_uint16_t(uint16_t p0, uint16_t p1, uint16_t p2) { return mad(p0, p1, p2); }

// NATIVE_HALF: %dx.umad = call <2 x i16>  @llvm.dx.umad.v2i16(<2 x i16> %0, <2 x i16> %1, <2 x i16> %2)
// NATIVE_HALF: ret <2 x i16> %dx.umad
uint16_t2 test_mad_uint16_t2(uint16_t2 p0, uint16_t2 p1, uint16_t2 p2) { return mad(p0, p1, p2); }

// NATIVE_HALF: %dx.umad = call <3 x i16>  @llvm.dx.umad.v3i16(<3 x i16> %0, <3 x i16> %1, <3 x i16> %2)
// NATIVE_HALF: ret <3 x i16> %dx.umad
uint16_t3 test_mad_uint16_t3(uint16_t3 p0, uint16_t3 p1, uint16_t3 p2) { return mad(p0, p1, p2); }

// NATIVE_HALF: %dx.umad = call <4 x i16>  @llvm.dx.umad.v4i16(<4 x i16> %0, <4 x i16> %1, <4 x i16> %2)
// NATIVE_HALF: ret <4 x i16> %dx.umad
uint16_t4 test_mad_uint16_t4(uint16_t4 p0, uint16_t4 p1, uint16_t4 p2) { return mad(p0, p1, p2); }

// NATIVE_HALF: %dx.imad = call i16 @llvm.dx.imad.i16(i16 %0, i16 %1, i16 %2)
// NATIVE_HALF: ret i16 %dx.imad
int16_t test_mad_int16_t(int16_t p0, int16_t p1, int16_t p2) { return mad(p0, p1, p2); }

// NATIVE_HALF: %dx.imad = call <2 x i16>  @llvm.dx.imad.v2i16(<2 x i16> %0, <2 x i16> %1, <2 x i16> %2)
// NATIVE_HALF: ret <2 x i16> %dx.imad
int16_t2 test_mad_int16_t2(int16_t2 p0, int16_t2 p1, int16_t2 p2) { return mad(p0, p1, p2); }

// NATIVE_HALF: %dx.imad = call <3 x i16>  @llvm.dx.imad.v3i16(<3 x i16> %0, <3 x i16> %1, <3 x i16> %2)
// NATIVE_HALF: ret <3 x i16> %dx.imad
int16_t3 test_mad_int16_t3(int16_t3 p0, int16_t3 p1, int16_t3 p2) { return mad(p0, p1, p2); }

// NATIVE_HALF: %dx.imad = call <4 x i16>  @llvm.dx.imad.v4i16(<4 x i16> %0, <4 x i16> %1, <4 x i16> %2)
// NATIVE_HALF: ret <4 x i16> %dx.imad
int16_t4 test_mad_int16_t4(int16_t4 p0, int16_t4 p1, int16_t4 p2) { return mad(p0, p1, p2); }
#endif // __HLSL_ENABLE_16_BIT

// NATIVE_HALF: %dx.fmad = call half @llvm.fmuladd.f16(half %0, half %1, half %2)
// NATIVE_HALF: ret half %dx.fmad
// NO_HALF: %dx.fmad = call float @llvm.fmuladd.f32(float %0, float %1, float %2)
// NO_HALF: ret float %dx.fmad
half test_mad_half(half p0, half p1, half p2) { return mad(p0, p1, p2); }

// NATIVE_HALF: %dx.fmad = call <2 x half>  @llvm.fmuladd.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %2)
// NATIVE_HALF: ret <2 x half> %dx.fmad
// NO_HALF: %dx.fmad = call <2 x float>  @llvm.fmuladd.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2)
// NO_HALF: ret <2 x float> %dx.fmad
half2 test_mad_half2(half2 p0, half2 p1, half2 p2) { return mad(p0, p1, p2); }

// NATIVE_HALF: %dx.fmad = call <3 x half>  @llvm.fmuladd.v3f16(<3 x half> %0, <3 x half> %1, <3 x half> %2)
// NATIVE_HALF: ret <3 x half> %dx.fmad
// NO_HALF: %dx.fmad = call <3 x float>  @llvm.fmuladd.v3f32(<3 x float> %0, <3 x float> %1, <3 x float> %2)
// NO_HALF: ret <3 x float> %dx.fmad
half3 test_mad_half3(half3 p0, half3 p1, half3 p2) { return mad(p0, p1, p2); }

// NATIVE_HALF: %dx.fmad = call <4 x half>  @llvm.fmuladd.v4f16(<4 x half> %0, <4 x half> %1, <4 x half> %2)
// NATIVE_HALF: ret <4 x half> %dx.fmad
// NO_HALF: %dx.fmad = call <4 x float>  @llvm.fmuladd.v4f32(<4 x float> %0, <4 x float> %1, <4 x float> %2)
// NO_HALF: ret <4 x float> %dx.fmad
half4 test_mad_half4(half4 p0, half4 p1, half4 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.fmad = call float @llvm.fmuladd.f32(float %0, float %1, float %2)
// CHECK: ret float %dx.fmad
float test_mad_float(float p0, float p1, float p2) { return mad(p0, p1, p2); }

// CHECK: %dx.fmad = call <2 x float>  @llvm.fmuladd.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2)
// CHECK: ret <2 x float> %dx.fmad
float2 test_mad_float2(float2 p0, float2 p1, float2 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.fmad = call <3 x float>  @llvm.fmuladd.v3f32(<3 x float> %0, <3 x float> %1, <3 x float> %2)
// CHECK: ret <3 x float> %dx.fmad
float3 test_mad_float3(float3 p0, float3 p1, float3 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.fmad = call <4 x float>  @llvm.fmuladd.v4f32(<4 x float> %0, <4 x float> %1, <4 x float> %2)
// CHECK: ret <4 x float> %dx.fmad
float4 test_mad_float4(float4 p0, float4 p1, float4 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.fmad = call double @llvm.fmuladd.f64(double %0, double %1, double %2)
// CHECK: ret double %dx.fmad
double test_mad_double(double p0, double p1, double p2) { return mad(p0, p1, p2); }

// CHECK: %dx.fmad = call <2 x double>  @llvm.fmuladd.v2f64(<2 x double> %0, <2 x double> %1, <2 x double> %2)
// CHECK: ret <2 x double> %dx.fmad
double2 test_mad_double2(double2 p0, double2 p1, double2 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.fmad = call <3 x double>  @llvm.fmuladd.v3f64(<3 x double> %0, <3 x double> %1, <3 x double> %2)
// CHECK: ret <3 x double> %dx.fmad
double3 test_mad_double3(double3 p0, double3 p1, double3 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.fmad = call <4 x double>  @llvm.fmuladd.v4f64(<4 x double> %0, <4 x double> %1, <4 x double> %2)
// CHECK: ret <4 x double> %dx.fmad
double4 test_mad_double4(double4 p0, double4 p1, double4 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.imad = call i32 @llvm.dx.imad.i32(i32 %0, i32 %1, i32 %2)
// CHECK: ret i32 %dx.imad
int test_mad_int(int p0, int p1, int p2) { return mad(p0, p1, p2); }

// CHECK: %dx.imad = call <2 x i32>  @llvm.dx.imad.v2i32(<2 x i32> %0, <2 x i32> %1, <2 x i32> %2)
// CHECK: ret <2 x i32> %dx.imad
int2 test_mad_int2(int2 p0, int2 p1, int2 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.imad = call <3 x i32>  @llvm.dx.imad.v3i32(<3 x i32> %0, <3 x i32> %1, <3 x i32> %2)
// CHECK: ret <3 x i32> %dx.imad
int3 test_mad_int3(int3 p0, int3 p1, int3 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.imad = call <4 x i32>  @llvm.dx.imad.v4i32(<4 x i32> %0, <4 x i32> %1, <4 x i32> %2)
// CHECK: ret <4 x i32> %dx.imad
int4 test_mad_int4(int4 p0, int4 p1, int4 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.imad = call i64 @llvm.dx.imad.i64(i64 %0, i64 %1, i64 %2)
// CHECK: ret i64 %dx.imad
int64_t test_mad_int64_t(int64_t p0, int64_t p1, int64_t p2) { return mad(p0, p1, p2); }

// CHECK: %dx.imad = call <2 x i64>  @llvm.dx.imad.v2i64(<2 x i64> %0, <2 x i64> %1, <2 x i64> %2)
// CHECK: ret <2 x i64> %dx.imad
int64_t2 test_mad_int64_t2(int64_t2 p0, int64_t2 p1, int64_t2 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.imad = call <3 x i64>  @llvm.dx.imad.v3i64(<3 x i64> %0, <3 x i64> %1, <3 x i64> %2)
// CHECK: ret <3 x i64> %dx.imad
int64_t3 test_mad_int64_t3(int64_t3 p0, int64_t3 p1, int64_t3 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.imad = call <4 x i64>  @llvm.dx.imad.v4i64(<4 x i64> %0, <4 x i64> %1, <4 x i64> %2)
// CHECK: ret <4 x i64> %dx.imad
int64_t4 test_mad_int64_t4(int64_t4 p0, int64_t4 p1, int64_t4 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.umad = call i32 @llvm.dx.umad.i32(i32 %0, i32 %1, i32 %2)
// CHECK: ret i32 %dx.umad
uint test_mad_uint(uint p0, uint p1, uint p2) { return mad(p0, p1, p2); }

// CHECK: %dx.umad = call <2 x i32>  @llvm.dx.umad.v2i32(<2 x i32> %0, <2 x i32> %1, <2 x i32> %2)
// CHECK: ret <2 x i32> %dx.umad
uint2 test_mad_uint2(uint2 p0, uint2 p1, uint2 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.umad = call <3 x i32>  @llvm.dx.umad.v3i32(<3 x i32> %0, <3 x i32> %1, <3 x i32> %2)
// CHECK: ret <3 x i32> %dx.umad
uint3 test_mad_uint3(uint3 p0, uint3 p1, uint3 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.umad = call <4 x i32>  @llvm.dx.umad.v4i32(<4 x i32> %0, <4 x i32> %1, <4 x i32> %2)
// CHECK: ret <4 x i32> %dx.umad
uint4 test_mad_uint4(uint4 p0, uint4 p1, uint4 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.umad = call i64 @llvm.dx.umad.i64(i64 %0, i64 %1, i64 %2)
// CHECK: ret i64 %dx.umad
uint64_t test_mad_uint64_t(uint64_t p0, uint64_t p1, uint64_t p2) { return mad(p0, p1, p2); }

// CHECK: %dx.umad = call <2 x i64>  @llvm.dx.umad.v2i64(<2 x i64> %0, <2 x i64> %1, <2 x i64> %2)
// CHECK: ret <2 x i64> %dx.umad
uint64_t2 test_mad_uint64_t2(uint64_t2 p0, uint64_t2 p1, uint64_t2 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.umad = call <3 x i64>  @llvm.dx.umad.v3i64(<3 x i64> %0, <3 x i64> %1, <3 x i64> %2)
// CHECK: ret <3 x i64> %dx.umad
uint64_t3 test_mad_uint64_t3(uint64_t3 p0, uint64_t3 p1, uint64_t3 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.umad = call <4 x i64>  @llvm.dx.umad.v4i64(<4 x i64> %0, <4 x i64> %1, <4 x i64> %2)
// CHECK: ret <4 x i64> %dx.umad
uint64_t4 test_mad_uint64_t4(uint64_t4 p0, uint64_t4 p1, uint64_t4 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.fmad = call <2 x float>  @llvm.fmuladd.v2f32(<2 x float> %splat.splat, <2 x float> %1, <2 x float> %2)
// CHECK: ret <2 x float> %dx.fmad
float2 test_mad_float2_splat(float p0, float2 p1, float2 p2) { return mad(p0, p1, p2); }

// CHECK: %dx.fmad = call <3 x float>  @llvm.fmuladd.v3f32(<3 x float> %splat.splat, <3 x float> %1, <3 x float> %2)
// CHECK: ret <3 x float> %dx.fmad
float3 test_mad_float3_splat(float p0, float3 p1, float3 p2) { return mad(p0, p1, p2); }

// CHECK:  %dx.fmad = call <4 x float>  @llvm.fmuladd.v4f32(<4 x float> %splat.splat, <4 x float> %1, <4 x float> %2)
// CHECK:  ret <4 x float> %dx.fmad
float4 test_mad_float4_splat(float p0, float4 p1, float4 p2) { return mad(p0, p1, p2); }

// CHECK: %conv = sitofp i32 %2 to float
// CHECK: %splat.splatinsert = insertelement <2 x float> poison, float %conv, i64 0
// CHECK: %splat.splat = shufflevector <2 x float> %splat.splatinsert, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK: %dx.fmad = call <2 x float>  @llvm.fmuladd.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %splat.splat)
// CHECK: ret <2 x float> %dx.fmad
float2 test_mad_float2_int_splat(float2 p0, float2 p1, int p2) {
  return mad(p0, p1, p2);
}

// CHECK: %conv = sitofp i32 %2 to float
// CHECK: %splat.splatinsert = insertelement <3 x float> poison, float %conv, i64 0
// CHECK: %splat.splat = shufflevector <3 x float> %splat.splatinsert, <3 x float> poison, <3 x i32> zeroinitializer
// CHECK:  %dx.fmad = call <3 x float>  @llvm.fmuladd.v3f32(<3 x float> %0, <3 x float> %1, <3 x float> %splat.splat)
// CHECK: ret <3 x float> %dx.fmad
float3 test_mad_float3_int_splat(float3 p0, float3 p1, int p2) {
  return mad(p0, p1, p2);
}
