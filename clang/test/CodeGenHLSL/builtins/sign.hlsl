// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,DXIL_CHECK,DXIL_NATIVE_HALF,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,DXIL_CHECK,NO_HALF,DXIL_NO_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SPIR_CHECK,NATIVE_HALF,SPIR_NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,SPIR_CHECK,NO_HALF,SPIR_NO_HALF

// DXIL_NATIVE_HALF: define noundef i32 @
// SPIR_NATIVE_HALF: define spir_func noundef i32 @
// DXIL_NATIVE_HALF: %hlsl.sign = call i32 @llvm.dx.sign.f16(
// SPIR_NATIVE_HALF: %hlsl.sign = call i32 @llvm.spv.sign.f16(
// NATIVE_HALF: ret i32 %hlsl.sign
// DXIL_NO_HALF: define noundef i32 @
// SPIR_NO_HALF: define spir_func noundef i32 @
// DXIL_NO_HALF: %hlsl.sign = call i32 @llvm.dx.sign.f32(
// SPIR_NO_HALF: %hlsl.sign = call i32 @llvm.spv.sign.f32(
// NO_HALF: ret i32 %hlsl.sign
int test_sign_half(half p0) { return sign(p0); }

// DXIL_NATIVE_HALF: define noundef <2 x i32> @
// SPIR_NATIVE_HALF: define spir_func noundef <2 x i32> @
// DXIL_NATIVE_HALF: %hlsl.sign = call <2 x i32> @llvm.dx.sign.v2f16(
// SPIR_NATIVE_HALF: %hlsl.sign = call <2 x i32> @llvm.spv.sign.v2f16(
// NATIVE_HALF: ret <2 x i32> %hlsl.sign
// DXIL_NO_HALF: define noundef <2 x i32> @
// SPIR_NO_HALF: define spir_func noundef <2 x i32> @
// DXIL_NO_HALF: %hlsl.sign = call <2 x i32> @llvm.dx.sign.v2f32(
// SPIR_NO_HALF: %hlsl.sign = call <2 x i32> @llvm.spv.sign.v2f32(
// NO_HALF: ret <2 x i32> %hlsl.sign
int2 test_sign_half2(half2 p0) { return sign(p0); }

// DXIL_NATIVE_HALF: define noundef <3 x i32> @
// SPIR_NATIVE_HALF: define spir_func noundef <3 x i32> @
// DXIL_NATIVE_HALF: %hlsl.sign = call <3 x i32> @llvm.dx.sign.v3f16(
// SPIR_NATIVE_HALF: %hlsl.sign = call <3 x i32> @llvm.spv.sign.v3f16(
// NATIVE_HALF: ret <3 x i32> %hlsl.sign
// DXIL_NO_HALF: define noundef <3 x i32> @
// SPIR_NO_HALF: define spir_func noundef <3 x i32> @
// DXIL_NO_HALF: %hlsl.sign = call <3 x i32> @llvm.dx.sign.v3f32(
// SPIR_NO_HALF: %hlsl.sign = call <3 x i32> @llvm.spv.sign.v3f32(
// NO_HALF: ret <3 x i32> %hlsl.sign
int3 test_sign_half3(half3 p0) { return sign(p0); }

// DXIL_NATIVE_HALF: define noundef <4 x i32> @
// SPIR_NATIVE_HALF: define spir_func noundef <4 x i32> @
// DXIL_NATIVE_HALF: %hlsl.sign = call <4 x i32> @llvm.dx.sign.v4f16(
// SPIR_NATIVE_HALF: %hlsl.sign = call <4 x i32> @llvm.spv.sign.v4f16(
// NATIVE_HALF: ret <4 x i32> %hlsl.sign
// DXIL_NO_HALF: define noundef <4 x i32> @
// SPIR_NO_HALF: define spir_func noundef <4 x i32> @
// DXIL_NO_HALF: %hlsl.sign = call <4 x i32> @llvm.dx.sign.v4f32(
// SPIR_NO_HALF: %hlsl.sign = call <4 x i32> @llvm.spv.sign.v4f32(
// NO_HALF: ret <4 x i32> %hlsl.sign
int4 test_sign_half4(half4 p0) { return sign(p0); }


// DXIL_CHECK: define noundef i32 @
// SPIR_CHECK: define spir_func noundef i32 @
// DXIL_CHECK: %hlsl.sign = call i32 @llvm.dx.sign.f32(
// SPIR_CHECK: %hlsl.sign = call i32 @llvm.spv.sign.f32(
// CHECK: ret i32 %hlsl.sign
int test_sign_float(float p0) { return sign(p0); }

// DXIL_CHECK: define noundef <2 x i32> @
// SPIR_CHECK: define spir_func noundef <2 x i32> @
// DXIL_CHECK: %hlsl.sign = call <2 x i32> @llvm.dx.sign.v2f32(
// SPIR_CHECK: %hlsl.sign = call <2 x i32> @llvm.spv.sign.v2f32(
// CHECK: ret <2 x i32> %hlsl.sign
int2 test_sign_float2(float2 p0) { return sign(p0); }

// DXIL_CHECK: define noundef <3 x i32> @
// SPIR_CHECK: define spir_func noundef <3 x i32> @
// DXIL_CHECK: %hlsl.sign = call <3 x i32> @llvm.dx.sign.v3f32(
// SPIR_CHECK: %hlsl.sign = call <3 x i32> @llvm.spv.sign.v3f32(
// CHECK: ret <3 x i32> %hlsl.sign
int3 test_sign_float3(float3 p0) { return sign(p0); }

// DXIL_CHECK: define noundef <4 x i32> @
// SPIR_CHECK: define spir_func noundef <4 x i32> @
// DXIL_CHECK: %hlsl.sign = call <4 x i32> @llvm.dx.sign.v4f32(
// SPIR_CHECK: %hlsl.sign = call <4 x i32> @llvm.spv.sign.v4f32(
// CHECK: ret <4 x i32> %hlsl.sign
int4 test_sign_float4(float4 p0) { return sign(p0); }


// DXIL_CHECK: define noundef i32 @
// SPIR_CHECK: define spir_func noundef i32 @
// DXIL_CHECK: %hlsl.sign = call i32 @llvm.dx.sign.f64(
// SPIR_CHECK: %hlsl.sign = call i32 @llvm.spv.sign.f64(
// CHECK: ret i32 %hlsl.sign
int test_sign_double(double p0) { return sign(p0); }

// DXIL_CHECK: define noundef <2 x i32> @
// SPIR_CHECK: define spir_func noundef <2 x i32> @
// DXIL_CHECK: %hlsl.sign = call <2 x i32> @llvm.dx.sign.v2f64(
// SPIR_CHECK: %hlsl.sign = call <2 x i32> @llvm.spv.sign.v2f64(
// CHECK: ret <2 x i32> %hlsl.sign
int2 test_sign_double2(double2 p0) { return sign(p0); }

// DXIL_CHECK: define noundef <3 x i32> @
// SPIR_CHECK: define spir_func noundef <3 x i32> @
// DXIL_CHECK: %hlsl.sign = call <3 x i32> @llvm.dx.sign.v3f64(
// SPIR_CHECK: %hlsl.sign = call <3 x i32> @llvm.spv.sign.v3f64(
// CHECK: ret <3 x i32> %hlsl.sign
int3 test_sign_double3(double3 p0) { return sign(p0); }

// DXIL_CHECK: define noundef <4 x i32> @
// SPIR_CHECK: define spir_func noundef <4 x i32> @
// DXIL_CHECK: %hlsl.sign = call <4 x i32> @llvm.dx.sign.v4f64(
// SPIR_CHECK: %hlsl.sign = call <4 x i32> @llvm.spv.sign.v4f64(
// CHECK: ret <4 x i32> %hlsl.sign
int4 test_sign_double4(double4 p0) { return sign(p0); }


#ifdef __HLSL_ENABLE_16_BIT
// DXIL_NATIVE_HALF: define noundef i32 @
// SPIR_NATIVE_HALF: define spir_func noundef i32 @
// DXIL_NATIVE_HALF: %hlsl.sign = call i32 @llvm.dx.sign.i16(
// SPIR_NATIVE_HALF: %hlsl.sign = call i32 @llvm.spv.sign.i16(
// NATIVE_HALF: ret i32 %hlsl.sign
int test_sign_int16_t(int16_t p0) { return sign(p0); }

// DXIL_NATIVE_HALF: define noundef <2 x i32> @
// SPIR_NATIVE_HALF: define spir_func noundef <2 x i32> @
// DXIL_NATIVE_HALF: %hlsl.sign = call <2 x i32> @llvm.dx.sign.v2i16(
// SPIR_NATIVE_HALF: %hlsl.sign = call <2 x i32> @llvm.spv.sign.v2i16(
// NATIVE_HALF: ret <2 x i32> %hlsl.sign
int2 test_sign_int16_t2(int16_t2 p0) { return sign(p0); }

// DXIL_NATIVE_HALF: define noundef <3 x i32> @
// SPIR_NATIVE_HALF: define spir_func noundef <3 x i32> @
// DXIL_NATIVE_HALF: %hlsl.sign = call <3 x i32> @llvm.dx.sign.v3i16(
// SPIR_NATIVE_HALF: %hlsl.sign = call <3 x i32> @llvm.spv.sign.v3i16(
// NATIVE_HALF: ret <3 x i32> %hlsl.sign
int3 test_sign_int16_t3(int16_t3 p0) { return sign(p0); }

// DXIL_NATIVE_HALF: define noundef <4 x i32> @
// SPIR_NATIVE_HALF: define spir_func noundef <4 x i32> @
// DXIL_NATIVE_HALF: %hlsl.sign = call <4 x i32> @llvm.dx.sign.v4i16(
// SPIR_NATIVE_HALF: %hlsl.sign = call <4 x i32> @llvm.spv.sign.v4i16(
// NATIVE_HALF: ret <4 x i32> %hlsl.sign
int4 test_sign_int16_t4(int16_t4 p0) { return sign(p0); }
#endif // __HLSL_ENABLE_16_BIT


// DXIL_CHECK: define noundef i32 @
// SPIR_CHECK: define spir_func noundef i32 @
// DXIL_CHECK: %hlsl.sign = call i32 @llvm.dx.sign.i32(
// SPIR_CHECK: %hlsl.sign = call i32 @llvm.spv.sign.i32(
// CHECK: ret i32 %hlsl.sign
int test_sign_int(int p0) { return sign(p0); }

// DXIL_CHECK: define noundef <2 x i32> @
// SPIR_CHECK: define spir_func noundef <2 x i32> @
// DXIL_CHECK: %hlsl.sign = call <2 x i32> @llvm.dx.sign.v2i32(
// SPIR_CHECK: %hlsl.sign = call <2 x i32> @llvm.spv.sign.v2i32(
// CHECK: ret <2 x i32> %hlsl.sign
int2 test_sign_int2(int2 p0) { return sign(p0); }

// DXIL_CHECK: define noundef <3 x i32> @
// SPIR_CHECK: define spir_func noundef <3 x i32> @
// DXIL_CHECK: %hlsl.sign = call <3 x i32> @llvm.dx.sign.v3i32(
// SPIR_CHECK: %hlsl.sign = call <3 x i32> @llvm.spv.sign.v3i32(
// CHECK: ret <3 x i32> %hlsl.sign
int3 test_sign_int3(int3 p0) { return sign(p0); }

// DXIL_CHECK: define noundef <4 x i32> @
// SPIR_CHECK: define spir_func noundef <4 x i32> @
// DXIL_CHECK: %hlsl.sign = call <4 x i32> @llvm.dx.sign.v4i32(
// SPIR_CHECK: %hlsl.sign = call <4 x i32> @llvm.spv.sign.v4i32(
// CHECK: ret <4 x i32> %hlsl.sign
int4 test_sign_int4(int4 p0) { return sign(p0); }


// DXIL_CHECK: define noundef i32 @
// SPIR_CHECK: define spir_func noundef i32 @
// DXIL_CHECK: %hlsl.sign = call i32 @llvm.dx.sign.i64(
// SPIR_CHECK: %hlsl.sign = call i32 @llvm.spv.sign.i64(
// CHECK: ret i32 %hlsl.sign
int test_sign_int64_t(int64_t p0) { return sign(p0); }

// DXIL_CHECK: define noundef <2 x i32> @
// SPIR_CHECK: define spir_func noundef <2 x i32> @
// DXIL_CHECK: %hlsl.sign = call <2 x i32> @llvm.dx.sign.v2i64(
// SPIR_CHECK: %hlsl.sign = call <2 x i32> @llvm.spv.sign.v2i64(
// CHECK: ret <2 x i32> %hlsl.sign
int2 test_sign_int64_t2(int64_t2 p0) { return sign(p0); }

// DXIL_CHECK: define noundef <3 x i32> @
// SPIR_CHECK: define spir_func noundef <3 x i32> @
// DXIL_CHECK: %hlsl.sign = call <3 x i32> @llvm.dx.sign.v3i64(
// SPIR_CHECK: %hlsl.sign = call <3 x i32> @llvm.spv.sign.v3i64(
// CHECK: ret <3 x i32> %hlsl.sign
int3 test_sign_int64_t3(int64_t3 p0) { return sign(p0); }

// DXIL_CHECK: define noundef <4 x i32> @
// SPIR_CHECK: define spir_func noundef <4 x i32> @
// DXIL_CHECK: %hlsl.sign = call <4 x i32> @llvm.dx.sign.v4i64(
// SPIR_CHECK: %hlsl.sign = call <4 x i32> @llvm.spv.sign.v4i64(
// CHECK: ret <4 x i32> %hlsl.sign
int4 test_sign_int64_t4(int64_t4 p0) { return sign(p0); }
