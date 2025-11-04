// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s

// CHECK: define hidden noundef nofpclass(nan inf) float
// CHECK: %hlsl.f16tof32 = call reassoc nnan ninf nsz arcp afn float @llvm.dx.legacyf16tof32.i32(i32 %0)
// CHECK: ret float %hlsl.f16tof32
// CHECK: declare float @llvm.dx.legacyf16tof32.i32(i32)
float test_scalar(uint p0) { return __builtin_hlsl_elementwise_f16tof32(p0); }

// CHECK: define hidden noundef nofpclass(nan inf) <2 x float>
// CHECK: %hlsl.f16tof32 = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.dx.legacyf16tof32.v2i32(<2 x i32> %0)
// CHECK: ret <2 x float> %hlsl.f16tof32
// CHECK: declare <2 x float> @llvm.dx.legacyf16tof32.v2i32(<2 x i32>)
float2 test_uint2(uint2 p0) { return __builtin_hlsl_elementwise_f16tof32(p0); }

// CHECK: define hidden noundef nofpclass(nan inf) <3 x float> @_Z10test_uint3Dv3_j(<3 x i32> noundef %p0) #0 {
// CHECK: %hlsl.f16tof32 = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.dx.legacyf16tof32.v3i32(<3 x i32> %0)
// CHECK: ret <3 x float> %hlsl.f16tof32
// CHECK: declare <3 x float> @llvm.dx.legacyf16tof32.v3i32(<3 x i32>)
float3 test_uint3(uint3 p0) { return __builtin_hlsl_elementwise_f16tof32(p0); }

// CHECK: define hidden noundef nofpclass(nan inf) <4 x float> @_Z10test_uint4Dv4_j(<4 x i32> noundef %p0) #0 {
// CHECK: %hlsl.f16tof32 = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.legacyf16tof32.v4i32(<4 x i32> %0)
// CHECK: ret <4 x float> %hlsl.f16tof32
// CHECK: declare <4 x float> @llvm.dx.legacyf16tof32.v4i32(<4 x i32>)
float4 test_uint4(uint4 p0) { return __builtin_hlsl_elementwise_f16tof32(p0); }



