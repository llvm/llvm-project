// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl \
// RUN:   -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -O0 -o - | \
// RUN:   FileCheck %s
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl \
// RUN:   -triple spirv-unknown-vulkan-compute %s -emit-llvm -O0 -o - | \
// RUN:   FileCheck %s --check-prefix=SPIRV

// CHECK: define hidden noundef i32 @_Z11test_scalarf(float noundef nofpclass(nan inf) %p0) #0 {
// CHECK: %hlsl.f32tof16 = call i32 @llvm.dx.legacyf32tof16.f32(float %0)
// CHECK: ret i32 %hlsl.f32tof16
// CHECK: declare i32 @llvm.dx.legacyf32tof16.f32(float) #1
//
// SPIRV: define hidden spir_func noundef i32 @_Z11test_scalarf(float noundef nofpclass(nan inf) %p0) #0 {
// SPIRV: call i32 @llvm.spv.packhalf2x16.i32.v2f32(<2 x float> %[[#]])
// SPIRV-NOT: call i32 @llvm.spv.packhalf2x16.i32.v2f32(<2 x float> %[[#]])
// SPIRV: ret i32 %[[#]]
// SPIRV: declare i32 @llvm.spv.packhalf2x16.i32.v2f32(<2 x float>)
uint test_scalar(float p0) { return __builtin_hlsl_elementwise_f32tof16(p0); }

// CHECK: define hidden noundef <2 x i32> @_Z10test_uint2Dv2_f(<2 x float> noundef nofpclass(nan inf) %p0) #0 {
// CHECK: %hlsl.f32tof16 = call <2 x i32> @llvm.dx.legacyf32tof16.v2f32(<2 x float> %0)
// CHECK: ret <2 x i32> %hlsl.f32tof16
// CHECK: declare <2 x i32> @llvm.dx.legacyf32tof16.v2f32(<2 x float>) #1
//
// SPIRV: define hidden spir_func noundef <2 x i32> @_Z10test_uint2Dv2_f(<2 x float> noundef nofpclass(nan inf) %p0) #0 {
// SPIRV-COUNT-2: call i32 @llvm.spv.packhalf2x16.i32.v2f32(<2 x float> %[[#]])
// SPIRV-NOT: call i32 @llvm.spv.packhalf2x16.i32.v2f32(<2 x float> %[[#]])
// SPIRV: ret <2 x i32> %[[#]]
uint2 test_uint2(float2 p0) { return __builtin_hlsl_elementwise_f32tof16(p0); }

// CHECK: define hidden noundef <3 x i32> @_Z10test_uint3Dv3_f(<3 x float> noundef nofpclass(nan inf) %p0) #0 {
// CHECK: %hlsl.f32tof16 = call <3 x i32> @llvm.dx.legacyf32tof16.v3f32(<3 x float> %0)
// CHECK: ret <3 x i32> %hlsl.f32tof16
// CHECK: declare <3 x i32> @llvm.dx.legacyf32tof16.v3f32(<3 x float>) #1
//
// SPIRV: define hidden spir_func noundef <3 x i32> @_Z10test_uint3Dv3_f(<3 x float> noundef nofpclass(nan inf) %p0) #0 {
// SPIRV-COUNT-3: call i32 @llvm.spv.packhalf2x16.i32.v2f32(<2 x float> %[[#]])
// SPIRV-NOT: call i32 @llvm.spv.packhalf2x16.i32.v2f32(<2 x float> %[[#]])
// SPIRV: ret <3 x i32> %[[#]]
uint3 test_uint3(float3 p0) { return __builtin_hlsl_elementwise_f32tof16(p0); }

// CHECK: define hidden noundef <4 x i32> @_Z10test_uint4Dv4_f(<4 x float> noundef nofpclass(nan inf) %p0) #0 {
// CHECK: %hlsl.f32tof16 = call <4 x i32> @llvm.dx.legacyf32tof16.v4f32(<4 x float> %0)
// CHECK: ret <4 x i32> %hlsl.f32tof16
// CHECK: declare <4 x i32> @llvm.dx.legacyf32tof16.v4f32(<4 x float>) #1
//
// SPIRV: define hidden spir_func noundef <4 x i32> @_Z10test_uint4Dv4_f(<4 x float> noundef nofpclass(nan inf) %p0) #0 {
// SPIRV-COUNT-4: call i32 @llvm.spv.packhalf2x16.i32.v2f32(<2 x float> %[[#]])
// SPIRV-NOT: call i32 @llvm.spv.packhalf2x16.i32.v2f32(<2 x float> %[[#]])
// SPIRV: ret <4 x i32> %[[#]]
uint4 test_uint4(float4 p0) { return __builtin_hlsl_elementwise_f32tof16(p0); }
