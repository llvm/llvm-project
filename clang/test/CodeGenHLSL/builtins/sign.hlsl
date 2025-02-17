// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DTARGET=dx -DFNATTRS=noundef
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DTARGET=dx -DFNATTRS=noundef
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DTARGET=spv -DFNATTRS="spir_func noundef"
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DTARGET=spv -DFNATTRS="spir_func noundef"

// NATIVE_HALF: define [[FNATTRS]] i32 @
// NATIVE_HALF: %hlsl.sign = call i32 @llvm.[[TARGET]].sign.f16(
// NATIVE_HALF: ret i32 %hlsl.sign
// NO_HALF: define [[FNATTRS]] i32 @
// NO_HALF: %hlsl.sign = call i32 @llvm.[[TARGET]].sign.f32(
// NO_HALF: ret i32 %hlsl.sign
int test_sign_half(half p0) { return sign(p0); }

// NATIVE_HALF: define [[FNATTRS]] <2 x i32> @
// NATIVE_HALF: %hlsl.sign = call <2 x i32> @llvm.[[TARGET]].sign.v2f16(
// NATIVE_HALF: ret <2 x i32> %hlsl.sign
// NO_HALF: define [[FNATTRS]] <2 x i32> @
// NO_HALF: %hlsl.sign = call <2 x i32> @llvm.[[TARGET]].sign.v2f32(
// NO_HALF: ret <2 x i32> %hlsl.sign
int2 test_sign_half2(half2 p0) { return sign(p0); }

// NATIVE_HALF: define [[FNATTRS]] <3 x i32> @
// NATIVE_HALF: %hlsl.sign = call <3 x i32> @llvm.[[TARGET]].sign.v3f16(
// NATIVE_HALF: ret <3 x i32> %hlsl.sign
// NO_HALF: define [[FNATTRS]] <3 x i32> @
// NO_HALF: %hlsl.sign = call <3 x i32> @llvm.[[TARGET]].sign.v3f32(
// NO_HALF: ret <3 x i32> %hlsl.sign
int3 test_sign_half3(half3 p0) { return sign(p0); }

// NATIVE_HALF: define [[FNATTRS]] <4 x i32> @
// NATIVE_HALF: %hlsl.sign = call <4 x i32> @llvm.[[TARGET]].sign.v4f16(
// NATIVE_HALF: ret <4 x i32> %hlsl.sign
// NO_HALF: define [[FNATTRS]] <4 x i32> @
// NO_HALF: %hlsl.sign = call <4 x i32> @llvm.[[TARGET]].sign.v4f32(
// NO_HALF: ret <4 x i32> %hlsl.sign
int4 test_sign_half4(half4 p0) { return sign(p0); }


// CHECK: define [[FNATTRS]] i32 @
// CHECK: %hlsl.sign = call i32 @llvm.[[TARGET]].sign.f32(
// CHECK: ret i32 %hlsl.sign
int test_sign_float(float p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <2 x i32> @
// CHECK: %hlsl.sign = call <2 x i32> @llvm.[[TARGET]].sign.v2f32(
// CHECK: ret <2 x i32> %hlsl.sign
int2 test_sign_float2(float2 p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <3 x i32> @
// CHECK: %hlsl.sign = call <3 x i32> @llvm.[[TARGET]].sign.v3f32(
// CHECK: ret <3 x i32> %hlsl.sign
int3 test_sign_float3(float3 p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <4 x i32> @
// CHECK: %hlsl.sign = call <4 x i32> @llvm.[[TARGET]].sign.v4f32(
// CHECK: ret <4 x i32> %hlsl.sign
int4 test_sign_float4(float4 p0) { return sign(p0); }


// CHECK: define [[FNATTRS]] i32 @
// CHECK: %hlsl.sign = call i32 @llvm.[[TARGET]].sign.f64(
// CHECK: ret i32 %hlsl.sign
int test_sign_double(double p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <2 x i32> @
// CHECK: %hlsl.sign = call <2 x i32> @llvm.[[TARGET]].sign.v2f64(
// CHECK: ret <2 x i32> %hlsl.sign
int2 test_sign_double2(double2 p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <3 x i32> @
// CHECK: %hlsl.sign = call <3 x i32> @llvm.[[TARGET]].sign.v3f64(
// CHECK: ret <3 x i32> %hlsl.sign
int3 test_sign_double3(double3 p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <4 x i32> @
// CHECK: %hlsl.sign = call <4 x i32> @llvm.[[TARGET]].sign.v4f64(
// CHECK: ret <4 x i32> %hlsl.sign
int4 test_sign_double4(double4 p0) { return sign(p0); }


#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF: define [[FNATTRS]] i32 @
// NATIVE_HALF: %hlsl.sign = call i32 @llvm.[[TARGET]].sign.i16(
// NATIVE_HALF: ret i32 %hlsl.sign
int test_sign_int16_t(int16_t p0) { return sign(p0); }

// NATIVE_HALF: define [[FNATTRS]] <2 x i32> @
// NATIVE_HALF: %hlsl.sign = call <2 x i32> @llvm.[[TARGET]].sign.v2i16(
// NATIVE_HALF: ret <2 x i32> %hlsl.sign
int2 test_sign_int16_t2(int16_t2 p0) { return sign(p0); }

// NATIVE_HALF: define [[FNATTRS]] <3 x i32> @
// NATIVE_HALF: %hlsl.sign = call <3 x i32> @llvm.[[TARGET]].sign.v3i16(
// NATIVE_HALF: ret <3 x i32> %hlsl.sign
int3 test_sign_int16_t3(int16_t3 p0) { return sign(p0); }

// NATIVE_HALF: define [[FNATTRS]] <4 x i32> @
// NATIVE_HALF: %hlsl.sign = call <4 x i32> @llvm.[[TARGET]].sign.v4i16(
// NATIVE_HALF: ret <4 x i32> %hlsl.sign
int4 test_sign_int16_t4(int16_t4 p0) { return sign(p0); }


// NATIVE_HALF: define [[FNATTRS]] i32 @
// NATIVE_HALF: [[CMP:%.*]] = icmp eq i16 [[ARG:%.*]], 0
// NATIVE_HALF: %hlsl.sign = select i1 [[CMP]], i32 0, i32 1
int test_sign_uint16_t(uint16_t p0) { return sign(p0); }

// NATIVE_HALF: define [[FNATTRS]] <2 x i32> @
// NATIVE_HALF: [[CMP:%.*]] = icmp eq <2 x i16> [[ARG:%.*]], zeroinitializer
// NATIVE_HALF: %hlsl.sign = select <2 x i1> [[CMP]], <2 x i32> zeroinitializer, <2 x i32> splat (i32 1)
int2 test_sign_uint16_t2(uint16_t2 p0) { return sign(p0); }

// NATIVE_HALF: define [[FNATTRS]] <3 x i32> @
// NATIVE_HALF: [[CMP:%.*]] = icmp eq <3 x i16> [[ARG:%.*]], zeroinitializer
// NATIVE_HALF: %hlsl.sign = select <3 x i1> [[CMP]], <3 x i32> zeroinitializer, <3 x i32> splat (i32 1)
int3 test_sign_uint16_t3(uint16_t3 p0) { return sign(p0); }

// NATIVE_HALF: define [[FNATTRS]] <4 x i32> @
// NATIVE_HALF: [[CMP:%.*]] = icmp eq <4 x i16> [[ARG:%.*]], zeroinitializer
// NATIVE_HALF: %hlsl.sign = select <4 x i1> [[CMP]], <4 x i32> zeroinitializer, <4 x i32> splat (i32 1)
int4 test_sign_uint16_t4(uint16_t4 p0) { return sign(p0); }
#endif // __HLSL_ENABLE_16_BIT


// CHECK: define [[FNATTRS]] i32 @
// CHECK: %hlsl.sign = call i32 @llvm.[[TARGET]].sign.i32(
// CHECK: ret i32 %hlsl.sign
int test_sign_int(int p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <2 x i32> @
// CHECK: %hlsl.sign = call <2 x i32> @llvm.[[TARGET]].sign.v2i32(
// CHECK: ret <2 x i32> %hlsl.sign
int2 test_sign_int2(int2 p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <3 x i32> @
// CHECK: %hlsl.sign = call <3 x i32> @llvm.[[TARGET]].sign.v3i32(
// CHECK: ret <3 x i32> %hlsl.sign
int3 test_sign_int3(int3 p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <4 x i32> @
// CHECK: %hlsl.sign = call <4 x i32> @llvm.[[TARGET]].sign.v4i32(
// CHECK: ret <4 x i32> %hlsl.sign
int4 test_sign_int4(int4 p0) { return sign(p0); }


// CHECK: define [[FNATTRS]] i32 @
// CHECK: [[CMP:%.*]] = icmp eq i32 [[ARG:%.*]], 0
// CHECK: %hlsl.sign = select i1 [[CMP]], i32 0, i32 1
int test_sign_uint(uint p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <2 x i32> @
// CHECK: [[CMP:%.*]] = icmp eq <2 x i32> [[ARG:%.*]], zeroinitializer
// CHECK: %hlsl.sign = select <2 x i1> [[CMP]], <2 x i32> zeroinitializer, <2 x i32> splat (i32 1)
int2 test_sign_uint2(uint2 p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <3 x i32> @
// CHECK: [[CMP:%.*]] = icmp eq <3 x i32> [[ARG:%.*]], zeroinitializer
// CHECK: %hlsl.sign = select <3 x i1> [[CMP]], <3 x i32> zeroinitializer, <3 x i32> splat (i32 1)
int3 test_sign_uint3(uint3 p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <4 x i32> @
// CHECK: [[CMP:%.*]] = icmp eq <4 x i32> [[ARG:%.*]], zeroinitializer
// CHECK: %hlsl.sign = select <4 x i1> [[CMP]], <4 x i32> zeroinitializer, <4 x i32> splat (i32 1)
int4 test_sign_uint4(uint4 p0) { return sign(p0); }


// CHECK: define [[FNATTRS]] i32 @
// CHECK: %hlsl.sign = call i32 @llvm.[[TARGET]].sign.i64(
// CHECK: ret i32 %hlsl.sign
int test_sign_int64_t(int64_t p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <2 x i32> @
// CHECK: %hlsl.sign = call <2 x i32> @llvm.[[TARGET]].sign.v2i64(
// CHECK: ret <2 x i32> %hlsl.sign
int2 test_sign_int64_t2(int64_t2 p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <3 x i32> @
// CHECK: %hlsl.sign = call <3 x i32> @llvm.[[TARGET]].sign.v3i64(
// CHECK: ret <3 x i32> %hlsl.sign
int3 test_sign_int64_t3(int64_t3 p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <4 x i32> @
// CHECK: %hlsl.sign = call <4 x i32> @llvm.[[TARGET]].sign.v4i64(
// CHECK: ret <4 x i32> %hlsl.sign
int4 test_sign_int64_t4(int64_t4 p0) { return sign(p0); }


// CHECK: define [[FNATTRS]] i32 @
// CHECK: [[CMP:%.*]] = icmp eq i64 [[ARG:%.*]], 0
// CHECK: %hlsl.sign = select i1 [[CMP]], i32 0, i32 1
int test_sign_uint64_t(uint64_t p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <2 x i32> @
// CHECK: [[CMP:%.*]] = icmp eq <2 x i64> [[ARG:%.*]], zeroinitializer
// CHECK: %hlsl.sign = select <2 x i1> [[CMP]], <2 x i32> zeroinitializer, <2 x i32> splat (i32 1)
int2 test_sign_uint64_t2(uint64_t2 p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <3 x i32> @
// CHECK: [[CMP:%.*]] = icmp eq <3 x i64> [[ARG:%.*]], zeroinitializer
// CHECK: %hlsl.sign = select <3 x i1> [[CMP]], <3 x i32> zeroinitializer, <3 x i32> splat (i32 1)
int3 test_sign_uint64_t3(uint64_t3 p0) { return sign(p0); }

// CHECK: define [[FNATTRS]] <4 x i32> @
// CHECK: [[CMP:%.*]] = icmp eq <4 x i64> [[ARG:%.*]], zeroinitializer
// CHECK: %hlsl.sign = select <4 x i1> [[CMP]], <4 x i32> zeroinitializer, <4 x i32> splat (i32 1)
int4 test_sign_uint64_t4(uint64_t4 p0) { return sign(p0); }
