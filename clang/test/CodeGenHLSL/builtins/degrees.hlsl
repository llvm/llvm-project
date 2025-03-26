// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DFNATTRS="noundef nofpclass(nan inf)" -DTARGET=dx
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DFNATTRS="noundef nofpclass(nan inf)" -DTARGET=dx
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DFNATTRS="spir_func noundef nofpclass(nan inf)" -DTARGET=spv
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DFNATTRS="spir_func noundef nofpclass(nan inf)" -DTARGET=spv

// NATIVE_HALF: define [[FNATTRS]] half @
// NATIVE_HALF: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn half @llvm.[[TARGET]].degrees.f16(
// NATIVE_HALF: ret half %hlsl.degrees
// NO_HALF: define [[FNATTRS]] float @
// NO_HALF: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].degrees.f32(
// NO_HALF: ret float %hlsl.degrees
half test_degrees_half(half p0) { return degrees(p0); }
// NATIVE_HALF: define [[FNATTRS]] <2 x half> @
// NATIVE_HALF: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.[[TARGET]].degrees.v2f16
// NATIVE_HALF: ret <2 x half> %hlsl.degrees
// NO_HALF: define [[FNATTRS]] <2 x float> @
// NO_HALF: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].degrees.v2f32(
// NO_HALF: ret <2 x float> %hlsl.degrees
half2 test_degrees_half2(half2 p0) { return degrees(p0); }
// NATIVE_HALF: define [[FNATTRS]] <3 x half> @
// NATIVE_HALF: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.[[TARGET]].degrees.v3f16
// NATIVE_HALF: ret <3 x half> %hlsl.degrees
// NO_HALF: define [[FNATTRS]] <3 x float> @
// NO_HALF: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].degrees.v3f32(
// NO_HALF: ret <3 x float> %hlsl.degrees
half3 test_degrees_half3(half3 p0) { return degrees(p0); }
// NATIVE_HALF: define [[FNATTRS]] <4 x half> @
// NATIVE_HALF: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.[[TARGET]].degrees.v4f16
// NATIVE_HALF: ret <4 x half> %hlsl.degrees
// NO_HALF: define [[FNATTRS]] <4 x float> @
// NO_HALF: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].degrees.v4f32(
// NO_HALF: ret <4 x float> %hlsl.degrees
half4 test_degrees_half4(half4 p0) { return degrees(p0); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].degrees.f32(
// CHECK: ret float %hlsl.degrees
float test_degrees_float(float p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].degrees.v2f32
// CHECK: ret <2 x float> %hlsl.degrees
float2 test_degrees_float2(float2 p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].degrees.v3f32
// CHECK: ret <3 x float> %hlsl.degrees
float3 test_degrees_float3(float3 p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].degrees.v4f32
// CHECK: ret <4 x float> %hlsl.degrees
float4 test_degrees_float4(float4 p0) { return degrees(p0); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].degrees.f32(
// CHECK: ret float %hlsl.degrees
float test_degrees_double(double p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].degrees.v2f32
// CHECK: ret <2 x float> %hlsl.degrees
float2 test_degrees_double2(double2 p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].degrees.v3f32
// CHECK: ret <3 x float> %hlsl.degrees
float3 test_degrees_double3(double3 p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].degrees.v4f32
// CHECK: ret <4 x float> %hlsl.degrees
float4 test_degrees_double4(double4 p0) { return degrees(p0); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].degrees.f32(
// CHECK: ret float %hlsl.degrees
float test_degrees_int(int p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].degrees.v2f32
// CHECK: ret <2 x float> %hlsl.degrees
float2 test_degrees_int2(int2 p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].degrees.v3f32
// CHECK: ret <3 x float> %hlsl.degrees
float3 test_degrees_int3(int3 p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].degrees.v4f32
// CHECK: ret <4 x float> %hlsl.degrees
float4 test_degrees_int4(int4 p0) { return degrees(p0); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].degrees.f32(
// CHECK: ret float %hlsl.degrees
float test_degrees_uint(uint p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].degrees.v2f32
// CHECK: ret <2 x float> %hlsl.degrees
float2 test_degrees_uint2(uint2 p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].degrees.v3f32
// CHECK: ret <3 x float> %hlsl.degrees
float3 test_degrees_uint3(uint3 p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].degrees.v4f32
// CHECK: ret <4 x float> %hlsl.degrees
float4 test_degrees_uint4(uint4 p0) { return degrees(p0); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].degrees.f32(
// CHECK: ret float %hlsl.degrees
float test_degrees_int64_t(int64_t p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].degrees.v2f32
// CHECK: ret <2 x float> %hlsl.degrees
float2 test_degrees_int64_t2(int64_t2 p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].degrees.v3f32
// CHECK: ret <3 x float> %hlsl.degrees
float3 test_degrees_int64_t3(int64_t3 p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].degrees.v4f32
// CHECK: ret <4 x float> %hlsl.degrees
float4 test_degrees_int64_t4(int64_t4 p0) { return degrees(p0); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].degrees.f32(
// CHECK: ret float %hlsl.degrees
float test_degrees_uint64_t(uint64_t p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].degrees.v2f32
// CHECK: ret <2 x float> %hlsl.degrees
float2 test_degrees_uint64_t2(uint64_t2 p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].degrees.v3f32
// CHECK: ret <3 x float> %hlsl.degrees
float3 test_degrees_uint64_t3(uint64_t3 p0) { return degrees(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %hlsl.degrees = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].degrees.v4f32
// CHECK: ret <4 x float> %hlsl.degrees
float4 test_degrees_uint64_t4(uint64_t4 p0) { return degrees(p0); }
