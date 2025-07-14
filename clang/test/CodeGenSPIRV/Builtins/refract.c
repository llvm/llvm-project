// RUN: %clang_cc1 -O1 -triple spirv-pc-vulkan-compute %s -emit-llvm -o - | FileCheck %s

typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));

// CHECK-LABEL: define spir_func <2 x float> @test_refract_float2(
// CHECK-SAME: <2 x float> noundef [[I:%.*]], <2 x float> noundef [[N:%.*]], float noundef [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK:    [[SPV_REFRACT:%.*]] = tail call <2 x float> @llvm.spv.refract.v2f32.f32(<2 x float> [[I]], <2 x float> [[N]], float [[ETA]])
// CHECK-NEXT:    ret <2 x float> [[SPV_REFRACT]]
//
float2 test_refract_float2(float2 I, float2 N, float eta) { return __builtin_spirv_refract(I, N, eta); }

// CHECK-LABEL: define spir_func <3 x float> @test_refract_float3(
// CHECK-SAME: <3 x float> noundef [[I:%.*]], <3 x float> noundef [[N:%.*]], float noundef [[ETA:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[SPV_REFRACT:%.*]] = tail call <3 x float> @llvm.spv.refract.v3f32.f32(<3 x float> [[I]], <3 x float> [[N]], float [[ETA]])
// CHECK-NEXT:    ret <3 x float> [[SPV_REFRACT]]
//
float3 test_refract_float3(float3 I, float3 N, float eta) { return __builtin_spirv_refract(I, N, eta); }

// CHECK-LABEL: define spir_func <4 x float> @test_refract_float4(
// CHECK-SAME: <4 x float> noundef [[I:%.*]], <4 x float> noundef [[N:%.*]], float noundef [[ETA:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[SPV_REFRACT:%.*]] = tail call <4 x float> @llvm.spv.refract.v4f32.f32(<4 x float> [[I]], <4 x float> [[N]], float [[ETA]])
// CHECK-NEXT:    ret <4 x float> [[SPV_REFRACT]]
//
float4 test_refract_float4(float4 I, float4 N, float eta) { return __builtin_spirv_refract(I, N, eta); }
