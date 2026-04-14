// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,CHECK-NATIVE_HALF -DTARGET=dx -DCC=""
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NO_HALF -DTARGET=dx -DCC=""

// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,CHECK-NATIVE_HALF -DTARGET=spv -DCC="spir_func "
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NO_HALF -DTARGET=spv -DCC="spir_func " 

// CHECK: %[[RET:.*]] = call [[CC]]i32 @llvm.[[TARGET]].quad.read.across.x.i32(i32 %[[#]])
// CHECK: ret i32 %[[RET]]
int test_int(int expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]<2 x i32> @llvm.[[TARGET]].quad.read.across.x.v2i32(<2 x i32> %[[#]])
// CHECK: ret <2 x i32> %[[RET]]
int2 test_int2(int2 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]<3 x i32> @llvm.[[TARGET]].quad.read.across.x.v3i32(<3 x i32> %[[#]])
// CHECK: ret <3 x i32> %[[RET]]
int3 test_int3(int3 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]<4 x i32> @llvm.[[TARGET]].quad.read.across.x.v4i32(<4 x i32> %[[#]])
// CHECK: ret <4 x i32> %[[RET]]
int4 test_int4(int4 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]i32 @llvm.[[TARGET]].quad.read.across.x.i32(i32 %[[#]])
// CHECK: ret i32 %[[RET]]
uint test_uint(uint expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]<2 x i32> @llvm.[[TARGET]].quad.read.across.x.v2i32(<2 x i32> %[[#]])
// CHECK: ret <2 x i32> %[[RET]]
uint2 test_uint2(uint2 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]<3 x i32> @llvm.[[TARGET]].quad.read.across.x.v3i32(<3 x i32> %[[#]])
// CHECK: ret <3 x i32> %[[RET]]
uint3 test_uint3(uint3 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]<4 x i32> @llvm.[[TARGET]].quad.read.across.x.v4i32(<4 x i32> %[[#]])
// CHECK: ret <4 x i32> %[[RET]]
uint4 test_uint4(uint4 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]i64 @llvm.[[TARGET]].quad.read.across.x.i64(i64 %[[#]])
// CHECK: ret i64 %[[RET]]
int64_t test_int64_t(int64_t expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]<2 x i64> @llvm.[[TARGET]].quad.read.across.x.v2i64(<2 x i64> %[[#]])
// CHECK: ret <2 x i64> %[[RET]]
int64_t2 test_int64_t2(int64_t2 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]<3 x i64> @llvm.[[TARGET]].quad.read.across.x.v3i64(<3 x i64> %[[#]])
// CHECK: ret <3 x i64> %[[RET]]
int64_t3 test_int64_t3(int64_t3 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]<4 x i64> @llvm.[[TARGET]].quad.read.across.x.v4i64(<4 x i64> %[[#]])
// CHECK: ret <4 x i64> %[[RET]]
int64_t4 test_int64_t4(int64_t4 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]i64 @llvm.[[TARGET]].quad.read.across.x.i64(i64 %[[#]])
// CHECK: ret i64 %[[RET]]
uint64_t test_uint64_t(uint64_t expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]<2 x i64> @llvm.[[TARGET]].quad.read.across.x.v2i64(<2 x i64> %[[#]])
// CHECK: ret <2 x i64> %[[RET]]
uint64_t2 test_uint64_t2(uint64_t2 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]<3 x i64> @llvm.[[TARGET]].quad.read.across.x.v3i64(<3 x i64> %[[#]])
// CHECK: ret <3 x i64> %[[RET]]
uint64_t3 test_uint64_t3(uint64_t3 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call [[CC]]<4 x i64> @llvm.[[TARGET]].quad.read.across.x.v4i64(<4 x i64> %[[#]])
// CHECK: ret <4 x i64> %[[RET]]
uint64_t4 test_uint64_t4(uint64_t4 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]float @llvm.[[TARGET]].quad.read.across.x.f32(float %[[#]])
// CHECK: ret float %[[RET]]
float test_float(float expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]<2 x float> @llvm.[[TARGET]].quad.read.across.x.v2f32(<2 x float> %[[#]])
// CHECK: ret <2 x float> %[[RET]]
float2 test_float2(float2 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]<3 x float> @llvm.[[TARGET]].quad.read.across.x.v3f32(<3 x float> %[[#]])
// CHECK: ret <3 x float> %[[RET]]
float3 test_float3(float3 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]<4 x float> @llvm.[[TARGET]].quad.read.across.x.v4f32(<4 x float> %[[#]])
// CHECK: ret <4 x float> %[[RET]]
float4 test_float4(float4 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]double @llvm.[[TARGET]].quad.read.across.x.f64(double %[[#]])
// CHECK: ret double %[[RET]]
double test_double(double expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]<2 x double> @llvm.[[TARGET]].quad.read.across.x.v2f64(<2 x double> %[[#]])
// CHECK: ret <2 x double> %[[RET]]
double2 test_double2(double2 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]<3 x double> @llvm.[[TARGET]].quad.read.across.x.v3f64(<3 x double> %[[#]])
// CHECK: ret <3 x double> %[[RET]]
double3 test_double3(double3 expr) { return QuadReadAcrossX(expr); }

// CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]<4 x double> @llvm.[[TARGET]].quad.read.across.x.v4f64(<4 x double> %[[#]])
// CHECK: ret <4 x double> %[[RET]]
double4 test_double4(double4 expr) { return QuadReadAcrossX(expr); }

// CHECK-NATIVE_HALF: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]half @llvm.[[TARGET]].quad.read.across.x.f16(half %[[#]])
// CHECK-NATIVE_HALF: ret half %[[RET]]
// CHECK-NO_HALF: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]float @llvm.[[TARGET]].quad.read.across.x.f32(float %[[#]])
// CHECK-NO_HALF: ret float %[[RET]]
half test_half(half expr) { return QuadReadAcrossX(expr); }

// CHECK-NATIVE_HALF: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]<2 x half> @llvm.[[TARGET]].quad.read.across.x.v2f16(<2 x half> %[[#]])
// CHECK-NATIVE_HALF: ret <2 x half> %[[RET]]
// CHECK-NO_HALF: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]<2 x float> @llvm.[[TARGET]].quad.read.across.x.v2f32(<2 x float> %[[#]])
// CHECK-NO_HALF: ret <2 x float> %[[RET]]
half2 test_half2(half2 expr) { return QuadReadAcrossX(expr); }

// CHECK-NATIVE_HALF: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]<3 x half> @llvm.[[TARGET]].quad.read.across.x.v3f16(<3 x half> %[[#]])
// CHECK-NATIVE_HALF: ret <3 x half> %[[RET]]
// CHECK-NO_HALF: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]<3 x float> @llvm.[[TARGET]].quad.read.across.x.v3f32(<3 x float> %[[#]])
// CHECK-NO_HALF: ret <3 x float> %[[RET]]
half3 test_half3(half3 expr) { return QuadReadAcrossX(expr); }

// CHECK-NATIVE_HALF: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]<4 x half> @llvm.[[TARGET]].quad.read.across.x.v4f16(<4 x half> %[[#]])
// CHECK-NATIVE_HALF: ret <4 x half> %[[RET]]
// CHECK-NO_HALF: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[CC]]<4 x float> @llvm.[[TARGET]].quad.read.across.x.v4f32(<4 x float> %[[#]])
// CHECK-NO_HALF: ret <4 x float> %[[RET]]
half4 test_half4(half4 expr) { return QuadReadAcrossX(expr); }

#ifdef __HLSL_ENABLE_16_BIT
// CHECK-NATIVE_HALF: %[[RET:.*]] = call [[CC]]i16 @llvm.[[TARGET]].quad.read.across.x.i16(i16 %[[#]])
// CHECK-NATIVE_HALF: ret i16 %[[RET]]
int16_t test_int16_t(int16_t expr) { return QuadReadAcrossX(expr); }

// CHECK-NATIVE_HALF: %[[RET:.*]] = call [[CC]]<2 x i16> @llvm.[[TARGET]].quad.read.across.x.v2i16(<2 x i16> %[[#]])
// CHECK-NATIVE_HALF: ret <2 x i16> %[[RET]]
int16_t2 test_int16_t2(int16_t2 expr) { return QuadReadAcrossX(expr); }

// CHECK-NATIVE_HALF: %[[RET:.*]] = call [[CC]]<3 x i16> @llvm.[[TARGET]].quad.read.across.x.v3i16(<3 x i16> %[[#]])
// CHECK-NATIVE_HALF: ret <3 x i16> %[[RET]]
int16_t3 test_int16_t3(int16_t3 expr) { return QuadReadAcrossX(expr); }

// CHECK-NATIVE_HALF: %[[RET:.*]] = call [[CC]]<4 x i16> @llvm.[[TARGET]].quad.read.across.x.v4i16(<4 x i16> %[[#]])
// CHECK-NATIVE_HALF: ret <4 x i16> %[[RET]]
int16_t4 test_int16_t4(int16_t4 expr) { return QuadReadAcrossX(expr); }

// CHECK-NATIVE_HALF: %[[RET:.*]] = call [[CC]]i16 @llvm.[[TARGET]].quad.read.across.x.i16(i16 %[[#]])
// CHECK-NATIVE_HALF: ret i16 %[[RET]]
uint16_t test_uint16_t(uint16_t expr) { return QuadReadAcrossX(expr); }

// CHECK-NATIVE_HALF: %[[RET:.*]] = call [[CC]]<2 x i16> @llvm.[[TARGET]].quad.read.across.x.v2i16(<2 x i16> %[[#]])
// CHECK-NATIVE_HALF: ret <2 x i16> %[[RET]]
uint16_t2 test_uint16_t2(uint16_t2 expr) { return QuadReadAcrossX(expr); }

// CHECK-NATIVE_HALF: %[[RET:.*]] = call [[CC]]<3 x i16> @llvm.[[TARGET]].quad.read.across.x.v3i16(<3 x i16> %[[#]])
// CHECK-NATIVE_HALF: ret <3 x i16> %[[RET]]
uint16_t3 test_uint16_t3(uint16_t3 expr) { return QuadReadAcrossX(expr); }

// CHECK-NATIVE_HALF: %[[RET:.*]] = call [[CC]]<4 x i16> @llvm.[[TARGET]].quad.read.across.x.v4i16(<4 x i16> %[[#]])
// CHECK-NATIVE_HALF: ret <4 x i16> %[[RET]]
uint16_t4 test_uint16_t4(uint16_t4 expr) { return QuadReadAcrossX(expr); }
#endif
