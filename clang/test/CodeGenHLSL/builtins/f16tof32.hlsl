// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl \
// RUN:   -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -O0 -o - | \
// RUN:   FileCheck %s
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl \
// RUN:   -triple spirv-unknown-vulkan-compute %s -emit-llvm -O0 -o - | \
// RUN:   FileCheck %s --check-prefix=SPIRV

// CHECK: define hidden noundef nofpclass(nan inf) float
// CHECK: %hlsl.f16tof32 = call reassoc nnan ninf nsz arcp afn float @llvm.dx.legacyf16tof32.i32(i32 %0)
// CHECK: ret float %hlsl.f16tof32
// CHECK: declare float @llvm.dx.legacyf16tof32.i32(i32)
//
// SPIRV: define hidden spir_func noundef nofpclass(nan inf) float @_Z11test_scalarj(i32 noundef %p0) #0 {
// SPIRV: %[[U1:[0-9]+]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.unpackhalf2x16.v2f32(i32 %[[#]])
// SPIRV: %[[R1:[0-9]+]] = extractelement <2 x float> %[[U1]], i64 0
// SPIRV: ret float %[[R1]]
// SPIRV: declare <2 x float> @llvm.spv.unpackhalf2x16.v2f32(i32) #2
float test_scalar(uint p0) { return f16tof32(p0); }

// CHECK: define hidden noundef nofpclass(nan inf) <2 x float>
// CHECK: %hlsl.f16tof32 = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.dx.legacyf16tof32.v2i32(<2 x i32> %0)
// CHECK: ret <2 x float> %hlsl.f16tof32
// CHECK: declare <2 x float> @llvm.dx.legacyf16tof32.v2i32(<2 x i32>)
//
// SPIRV: define hidden spir_func noundef nofpclass(nan inf) <2 x float> @_Z10test_uint2Dv2_j(<2 x i32> noundef %p0) #0 {
// SPRIV-COUNT-2: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.unpackhalf2x16.v2f32(i32 %[[#]]
// SPRIV-NOT: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.unpackhalf2x16.v2f32(i32 %[[#]])
// SPIRV: ret <2 x float> %9
float2 test_uint2(uint2 p0) { return f16tof32(p0); }

// CHECK: define hidden noundef nofpclass(nan inf) <3 x float> @_Z10test_uint3Dv3_j(<3 x i32> noundef %p0) #0 {
// CHECK: %hlsl.f16tof32 = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.dx.legacyf16tof32.v3i32(<3 x i32> %0)
// CHECK: ret <3 x float> %hlsl.f16tof32
// CHECK: declare <3 x float> @llvm.dx.legacyf16tof32.v3i32(<3 x i32>)
//
// SPIRV: define hidden spir_func noundef nofpclass(nan inf) <3 x float> @_Z10test_uint3Dv3_j(<3 x i32> noundef %p0) #0 {
// SPRIV-COUNT-3: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.unpackhalf2x16.v2f32(i32 %[[#]]
// SPRIV-NOT: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.unpackhalf2x16.v2f32(i32 %[[#]])
// SPIRV: ret <3 x float> %13
float3 test_uint3(uint3 p0) { return f16tof32(p0); }

// CHECK: define hidden noundef nofpclass(nan inf) <4 x float> @_Z10test_uint4Dv4_j(<4 x i32> noundef %p0) #0 {
// CHECK: %hlsl.f16tof32 = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.legacyf16tof32.v4i32(<4 x i32> %0)
// CHECK: ret <4 x float> %hlsl.f16tof32
// CHECK: declare <4 x float> @llvm.dx.legacyf16tof32.v4i32(<4 x i32>)
//
// SPRIV: define hidden spir_func noundef nofpclass(nan inf) <4 x float> @_Z10test_uint4Dv4_j(<4 x i32> noundef %p0) #0 {
// SPRIV-COUNT-4: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.unpackhalf2x16.v2f32(i32 %[[#]]
// SPRIV-NOT: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.unpackhalf2x16.v2f32(i32 %[[#]])
// SPRIV: ret <4 x float> %[[#]]
float4 test_uint4(uint4 p0) { return f16tof32(p0); }
