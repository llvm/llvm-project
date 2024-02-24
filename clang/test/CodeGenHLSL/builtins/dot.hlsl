// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF: %dx.dot = mul i16 %0, %1
// NATIVE_HALF: ret i16 %dx.dot
int16_t test_dot_short ( int16_t p0, int16_t p1 ) {
  return dot ( p0, p1 );
}

// NATIVE_HALF: %dx.dot = call i16 @llvm.dx.dot.v2i16(<2 x i16> %0, <2 x i16> %1)
// NATIVE_HALF: ret i16 %dx.dot
int16_t test_dot_short2 ( int16_t2 p0, int16_t2 p1 ) {
  return dot ( p0, p1 );
}

// NATIVE_HALF: %dx.dot = call i16 @llvm.dx.dot.v3i16(<3 x i16> %0, <3 x i16> %1)
// NATIVE_HALF: ret i16 %dx.dot
int16_t test_dot_short3 ( int16_t3 p0, int16_t3 p1 ) {
  return dot ( p0, p1 );
}

// NATIVE_HALF: %dx.dot = call i16 @llvm.dx.dot.v4i16(<4 x i16> %0, <4 x i16> %1)
// NATIVE_HALF: ret i16 %dx.dot
int16_t test_dot_short4 ( int16_t4 p0, int16_t4 p1 ) {
  return dot ( p0, p1 );
}

// NATIVE_HALF: %dx.dot = mul i16 %0, %1
// NATIVE_HALF: ret i16 %dx.dot
uint16_t test_dot_ushort ( uint16_t p0, uint16_t p1 ) {
  return dot ( p0, p1 );
}

// NATIVE_HALF: %dx.dot = call i16 @llvm.dx.dot.v2i16(<2 x i16> %0, <2 x i16> %1)
// NATIVE_HALF: ret i16 %dx.dot
uint16_t test_dot_ushort2 ( uint16_t2 p0, uint16_t2 p1 ) {
  return dot ( p0, p1 );
}

// NATIVE_HALF: %dx.dot = call i16 @llvm.dx.dot.v3i16(<3 x i16> %0, <3 x i16> %1)
// NATIVE_HALF: ret i16 %dx.dot
uint16_t test_dot_ushort3 ( uint16_t3 p0, uint16_t3 p1 ) {
  return dot ( p0, p1 );
}

// NATIVE_HALF: %dx.dot = call i16 @llvm.dx.dot.v4i16(<4 x i16> %0, <4 x i16> %1)
// NATIVE_HALF: ret i16 %dx.dot
uint16_t test_dot_ushort4 ( uint16_t4 p0, uint16_t4 p1 ) {
  return dot ( p0, p1 );
}
#endif

// CHECK: %dx.dot = mul i32 %0, %1
// CHECK: ret i32 %dx.dot
int test_dot_int ( int p0, int p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i32 @llvm.dx.dot.v2i32(<2 x i32> %0, <2 x i32> %1)
// CHECK: ret i32 %dx.dot
int test_dot_int2 ( int2 p0, int2 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i32 @llvm.dx.dot.v3i32(<3 x i32> %0, <3 x i32> %1)
// CHECK: ret i32 %dx.dot
int test_dot_int3 ( int3 p0, int3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i32 @llvm.dx.dot.v4i32(<4 x i32> %0, <4 x i32> %1)
// CHECK: ret i32 %dx.dot
int test_dot_int4 ( int4 p0, int4 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = mul i32 %0, %1
// CHECK: ret i32 %dx.dot
uint test_dot_uint ( uint p0, uint p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i32 @llvm.dx.dot.v2i32(<2 x i32> %0, <2 x i32> %1)
// CHECK: ret i32 %dx.dot
uint test_dot_uint2 ( uint2 p0, uint2 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i32 @llvm.dx.dot.v3i32(<3 x i32> %0, <3 x i32> %1)
// CHECK: ret i32 %dx.dot
uint test_dot_uint3 ( uint3 p0, uint3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i32 @llvm.dx.dot.v4i32(<4 x i32> %0, <4 x i32> %1)
// CHECK: ret i32 %dx.dot
uint test_dot_uint4 ( uint4 p0, uint4 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = mul i64 %0, %1
// CHECK: ret i64 %dx.dot
int64_t test_dot_long ( int64_t p0, int64_t p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i64 @llvm.dx.dot.v2i64(<2 x i64> %0, <2 x i64> %1)
// CHECK: ret i64 %dx.dot
int64_t test_dot_long2 ( int64_t2 p0, int64_t2 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i64 @llvm.dx.dot.v3i64(<3 x i64> %0, <3 x i64> %1)
// CHECK: ret i64 %dx.dot
int64_t test_dot_long3 ( int64_t3 p0, int64_t3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i64 @llvm.dx.dot.v4i64(<4 x i64> %0, <4 x i64> %1)
// CHECK: ret i64 %dx.dot
int64_t test_dot_long4 ( int64_t4 p0, int64_t4 p1 ) {
  return dot ( p0, p1 );
}

// CHECK:  %dx.dot = mul i64 %0, %1
// CHECK: ret i64 %dx.dot
uint64_t test_dot_ulong ( uint64_t p0, uint64_t p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i64 @llvm.dx.dot.v2i64(<2 x i64> %0, <2 x i64> %1)
// CHECK: ret i64 %dx.dot
uint64_t test_dot_ulong2 ( uint64_t2 p0, uint64_t2 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i64 @llvm.dx.dot.v3i64(<3 x i64> %0, <3 x i64> %1)
// CHECK: ret i64 %dx.dot
uint64_t test_dot_ulong3 ( uint64_t3 p0, uint64_t3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call i64 @llvm.dx.dot.v4i64(<4 x i64> %0, <4 x i64> %1)
// CHECK: ret i64 %dx.dot
uint64_t test_dot_ulong4 ( uint64_t4 p0, uint64_t4 p1 ) {
  return dot ( p0, p1 );
}

// NATIVE_HALF: %dx.dot = fmul half %0, %1
// NATIVE_HALF: ret half %dx.dot
// NO_HALF: %dx.dot = fmul float %0, %1
// NO_HALF: ret float %dx.dot
half test_dot_half ( half p0, half p1 ) {
  return dot ( p0, p1 );
}

// NATIVE_HALF: %dx.dot = call half @llvm.dx.dot.v2f16(<2 x half> %0, <2 x half> %1)
// NATIVE_HALF: ret half %dx.dot
// NO_HALF: %dx.dot = call float @llvm.dx.dot.v2f32(<2 x float> %0, <2 x float> %1)
// NO_HALF: ret float %dx.dot
half test_dot_half2 ( half2 p0, half2 p1 ) {
  return dot ( p0, p1 );
}

// NATIVE_HALF: %dx.dot = call half @llvm.dx.dot.v3f16(<3 x half> %0, <3 x half> %1)
// NATIVE_HALF: ret half %dx.dot
// NO_HALF: %dx.dot = call float @llvm.dx.dot.v3f32(<3 x float> %0, <3 x float> %1)
// NO_HALF: ret float %dx.dot
half test_dot_half3 ( half3 p0, half3 p1 ) {
  return dot ( p0, p1 );
}

// NATIVE_HALF: %dx.dot = call half @llvm.dx.dot.v4f16(<4 x half> %0, <4 x half> %1)
// NATIVE_HALF: ret half %dx.dot
// NO_HALF: %dx.dot = call float @llvm.dx.dot.v4f32(<4 x float> %0, <4 x float> %1)
// NO_HALF: ret float %dx.dot
half test_dot_half4 ( half4 p0, half4 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = fmul float %0, %1
// CHECK: ret float %dx.dot
float test_dot_float ( float p0, float p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call float @llvm.dx.dot.v2f32(<2 x float> %0, <2 x float> %1)
// CHECK: ret float %dx.dot
float test_dot_float2 ( float2 p0, float2 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call float @llvm.dx.dot.v3f32(<3 x float> %0, <3 x float> %1)
// CHECK: ret float %dx.dot
float test_dot_float3 ( float3 p0, float3 p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = call float @llvm.dx.dot.v4f32(<4 x float> %0, <4 x float> %1)
// CHECK: ret float %dx.dot
float test_dot_float4 ( float4 p0, float4 p1) {
  return dot ( p0, p1 );
}

// CHECK:  %dx.dot = call float @llvm.dx.dot.v2f32(<2 x float> %splat.splat, <2 x float> %1)
// CHECK: ret float %dx.dot
float test_dot_float2_splat ( float p0, float2 p1 ) {
  return dot( p0, p1 );
}

// CHECK:  %dx.dot = call float @llvm.dx.dot.v3f32(<3 x float> %splat.splat, <3 x float> %1)
// CHECK: ret float %dx.dot
float test_dot_float3_splat ( float p0, float3 p1 ) {
  return dot( p0, p1 );
}

// CHECK:  %dx.dot = call float @llvm.dx.dot.v4f32(<4 x float> %splat.splat, <4 x float> %1)
// CHECK: ret float %dx.dot
float test_dot_float4_splat ( float p0, float4 p1 ) {
  return dot( p0, p1 );
}

// CHECK: %conv = sitofp i32 %1 to float
// CHECK: %splat.splatinsert = insertelement <2 x float> poison, float %conv, i64 0
// CHECK: %splat.splat = shufflevector <2 x float> %splat.splatinsert, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK: %dx.dot = call float @llvm.dx.dot.v2f32(<2 x float> %0, <2 x float> %splat.splat)
// CHECK: ret float %dx.dot
float test_builtin_dot_float2_int_splat ( float2 p0, int p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %conv = sitofp i32 %1 to float
// CHECK: %splat.splatinsert = insertelement <3 x float> poison, float %conv, i64 0
// CHECK: %splat.splat = shufflevector <3 x float> %splat.splatinsert, <3 x float> poison, <3 x i32> zeroinitializer
// CHECK: %dx.dot = call float @llvm.dx.dot.v3f32(<3 x float> %0, <3 x float> %splat.splat)
// CHECK: ret float %dx.dot
float test_builtin_dot_float3_int_splat ( float3 p0, int p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %dx.dot = fmul double %0, %1
// CHECK: ret double %dx.dot
double test_dot_double ( double p0, double p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %conv = zext i1 %tobool to i32
// CHECK: %dx.dot = mul i32 %conv, %1
// CHECK: ret i32 %dx.dot
int test_dot_bool_scalar_arg0_type_promotion ( bool p0, int p1 ) {
  return dot ( p0, p1 );
}

// CHECK: %conv = zext i1 %tobool to i32
// CHECK: %dx.dot = mul i32 %0, %conv
// CHECK: ret i32 %dx.dot
int test_dot_bool_scalar_arg1_type_promotion ( int p0, bool p1 ) {
  return dot ( p0, p1 );
}
