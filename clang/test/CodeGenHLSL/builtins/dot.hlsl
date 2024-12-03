// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,DXCHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,DXCHECK,NO_HALF

// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SPVCHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,SPVCHECK,NO_HALF


// CHECK: %hlsl.dot = mul i32
// CHECK: ret i32 %hlsl.dot
int test_dot_int(int p0, int p1) { return dot(p0, p1); }

// Capture the expected interchange format so not every check needs to be duplicated
// DXCHECK: %hlsl.dot = call i32 @llvm.[[ICF:dx]].sdot.v2i32(<2 x i32>
// SPVCHECK: %hlsl.dot = call i32 @llvm.[[ICF:spv]].sdot.v2i32(<2 x i32>
// CHECK: ret i32 %hlsl.dot
int test_dot_int2(int2 p0, int2 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call i32 @llvm.[[ICF]].sdot.v3i32(<3 x i32>
// CHECK: ret i32 %hlsl.dot
int test_dot_int3(int3 p0, int3 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call i32 @llvm.[[ICF]].sdot.v4i32(<4 x i32>
// CHECK: ret i32 %hlsl.dot
int test_dot_int4(int4 p0, int4 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = mul i32
// CHECK: ret i32 %hlsl.dot
uint test_dot_uint(uint p0, uint p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call i32 @llvm.[[ICF]].udot.v2i32(<2 x i32>
// CHECK: ret i32 %hlsl.dot
uint test_dot_uint2(uint2 p0, uint2 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call i32 @llvm.[[ICF]].udot.v3i32(<3 x i32>
// CHECK: ret i32 %hlsl.dot
uint test_dot_uint3(uint3 p0, uint3 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call i32 @llvm.[[ICF]].udot.v4i32(<4 x i32>
// CHECK: ret i32 %hlsl.dot
uint test_dot_uint4(uint4 p0, uint4 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = mul i64
// CHECK: ret i64 %hlsl.dot
int64_t test_dot_long(int64_t p0, int64_t p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call i64 @llvm.[[ICF]].sdot.v2i64(<2 x i64>
// CHECK: ret i64 %hlsl.dot
int64_t test_dot_long2(int64_t2 p0, int64_t2 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call i64 @llvm.[[ICF]].sdot.v3i64(<3 x i64>
// CHECK: ret i64 %hlsl.dot
int64_t test_dot_long3(int64_t3 p0, int64_t3 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call i64 @llvm.[[ICF]].sdot.v4i64(<4 x i64>
// CHECK: ret i64 %hlsl.dot
int64_t test_dot_long4(int64_t4 p0, int64_t4 p1) { return dot(p0, p1); }

// CHECK:  %hlsl.dot = mul i64
// CHECK: ret i64 %hlsl.dot
uint64_t test_dot_ulong(uint64_t p0, uint64_t p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call i64 @llvm.[[ICF]].udot.v2i64(<2 x i64>
// CHECK: ret i64 %hlsl.dot
uint64_t test_dot_ulong2(uint64_t2 p0, uint64_t2 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call i64 @llvm.[[ICF]].udot.v3i64(<3 x i64>
// CHECK: ret i64 %hlsl.dot
uint64_t test_dot_ulong3(uint64_t3 p0, uint64_t3 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call i64 @llvm.[[ICF]].udot.v4i64(<4 x i64>
// CHECK: ret i64 %hlsl.dot
uint64_t test_dot_ulong4(uint64_t4 p0, uint64_t4 p1) { return dot(p0, p1); }

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF: %hlsl.dot = mul i16
// NATIVE_HALF: ret i16 %hlsl.dot
int16_t test_dot_short(int16_t p0, int16_t p1) { return dot(p0, p1); }

// NATIVE_HALF: %hlsl.dot = call i16 @llvm.[[ICF]].sdot.v2i16(<2 x i16>
// NATIVE_HALF: ret i16 %hlsl.dot
int16_t test_dot_short2(int16_t2 p0, int16_t2 p1) { return dot(p0, p1); }

// NATIVE_HALF: %hlsl.dot = call i16 @llvm.[[ICF]].sdot.v3i16(<3 x i16>
// NATIVE_HALF: ret i16 %hlsl.dot
int16_t test_dot_short3(int16_t3 p0, int16_t3 p1) { return dot(p0, p1); }

// NATIVE_HALF: %hlsl.dot = call i16 @llvm.[[ICF]].sdot.v4i16(<4 x i16>
// NATIVE_HALF: ret i16 %hlsl.dot
int16_t test_dot_short4(int16_t4 p0, int16_t4 p1) { return dot(p0, p1); }

// NATIVE_HALF: %hlsl.dot = mul i16
// NATIVE_HALF: ret i16 %hlsl.dot
uint16_t test_dot_ushort(uint16_t p0, uint16_t p1) { return dot(p0, p1); }

// NATIVE_HALF: %hlsl.dot = call i16 @llvm.[[ICF]].udot.v2i16(<2 x i16>
// NATIVE_HALF: ret i16 %hlsl.dot
uint16_t test_dot_ushort2(uint16_t2 p0, uint16_t2 p1) { return dot(p0, p1); }

// NATIVE_HALF: %hlsl.dot = call i16 @llvm.[[ICF]].udot.v3i16(<3 x i16>
// NATIVE_HALF: ret i16 %hlsl.dot
uint16_t test_dot_ushort3(uint16_t3 p0, uint16_t3 p1) { return dot(p0, p1); }

// NATIVE_HALF: %hlsl.dot = call i16 @llvm.[[ICF]].udot.v4i16(<4 x i16>
// NATIVE_HALF: ret i16 %hlsl.dot
uint16_t test_dot_ushort4(uint16_t4 p0, uint16_t4 p1) { return dot(p0, p1); }
#endif

// NATIVE_HALF: %hlsl.dot = fmul half
// NATIVE_HALF: ret half %hlsl.dot
// NO_HALF: %hlsl.dot = fmul float
// NO_HALF: ret float %hlsl.dot
half test_dot_half(half p0, half p1) { return dot(p0, p1); }

// NATIVE_HALF: %hlsl.dot = call half @llvm.[[ICF]].fdot.v2f16(<2 x half>
// NATIVE_HALF: ret half %hlsl.dot
// NO_HALF: %hlsl.dot = call float @llvm.[[ICF]].fdot.v2f32(<2 x float>
// NO_HALF: ret float %hlsl.dot
half test_dot_half2(half2 p0, half2 p1) { return dot(p0, p1); }

// NATIVE_HALF: %hlsl.dot = call half @llvm.[[ICF]].fdot.v3f16(<3 x half>
// NATIVE_HALF: ret half %hlsl.dot
// NO_HALF: %hlsl.dot = call float @llvm.[[ICF]].fdot.v3f32(<3 x float>
// NO_HALF: ret float %hlsl.dot
half test_dot_half3(half3 p0, half3 p1) { return dot(p0, p1); }

// NATIVE_HALF: %hlsl.dot = call half @llvm.[[ICF]].fdot.v4f16(<4 x half>
// NATIVE_HALF: ret half %hlsl.dot
// NO_HALF: %hlsl.dot = call float @llvm.[[ICF]].fdot.v4f32(<4 x float>
// NO_HALF: ret float %hlsl.dot
half test_dot_half4(half4 p0, half4 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = fmul float
// CHECK: ret float %hlsl.dot
float test_dot_float(float p0, float p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call float @llvm.[[ICF]].fdot.v2f32(<2 x float>
// CHECK: ret float %hlsl.dot
float test_dot_float2(float2 p0, float2 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call float @llvm.[[ICF]].fdot.v3f32(<3 x float>
// CHECK: ret float %hlsl.dot
float test_dot_float3(float3 p0, float3 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = call float @llvm.[[ICF]].fdot.v4f32(<4 x float>
// CHECK: ret float %hlsl.dot
float test_dot_float4(float4 p0, float4 p1) { return dot(p0, p1); }

// CHECK: %hlsl.dot = fmul double
// CHECK: ret double %hlsl.dot
double test_dot_double(double p0, double p1) { return dot(p0, p1); }
