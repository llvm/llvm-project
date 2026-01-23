// RUN: %clang_cc1 -O1 -triple spirv-pc-vulkan-compute %s -fnative-half-type -emit-llvm -o - | FileCheck %s

typedef _Float16 half;
typedef half half2 __attribute__((ext_vector_type(2)));
typedef half half3 __attribute__((ext_vector_type(3)));
typedef half half4 __attribute__((ext_vector_type(4)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));

// CHECK-LABEL: define spir_func half @test_refract_half(
// CHECK-SAME: half noundef [[I:%.*]], half noundef [[N:%.*]], half noundef [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK:    [[SPV_REFRACT:%.*]] = tail call half @llvm.spv.refract.f16.f16(half [[I]], half [[N]], half [[ETA]])
// CHECK-NEXT:    ret half [[SPV_REFRACT]]
//
half test_refract_half(half I, half N, half eta) { return __builtin_spirv_refract(I, N, eta); }

// CHECK-LABEL: define spir_func <2 x half> @test_refract_half2(
// CHECK-SAME: <2 x half> noundef [[I:%.*]], <2 x half> noundef [[N:%.*]], half noundef [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK:    [[SPV_REFRACT:%.*]] = tail call <2 x half> @llvm.spv.refract.v2f16.f16(<2 x half> [[I]], <2 x half> [[N]], half [[ETA]])
// CHECK-NEXT:    ret <2 x half> [[SPV_REFRACT]]
//
half2 test_refract_half2(half2 I, half2 N, half eta) { return __builtin_spirv_refract(I, N, eta); }

// CHECK-LABEL: define spir_func <3 x half> @test_refract_half3(
// CHECK-SAME: <3 x half> noundef [[I:%.*]], <3 x half> noundef [[N:%.*]], half noundef [[ETA:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[SPV_REFRACT:%.*]] = tail call <3 x half> @llvm.spv.refract.v3f16.f16(<3 x half> [[I]], <3 x half> [[N]], half [[ETA]])
// CHECK-NEXT:    ret <3 x half> [[SPV_REFRACT]]
//
half3 test_refract_half3(half3 I, half3 N, half eta) { return __builtin_spirv_refract(I, N, eta); }

// CHECK-LABEL: define spir_func <4 x half> @test_refract_half4(
// CHECK-SAME: <4 x half> noundef [[I:%.*]], <4 x half> noundef [[N:%.*]], half noundef [[ETA:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[SPV_REFRACT:%.*]] = tail call <4 x half> @llvm.spv.refract.v4f16.f16(<4 x half> [[I]], <4 x half> [[N]], half [[ETA]])
// CHECK-NEXT:    ret <4 x half> [[SPV_REFRACT]]
//
half4 test_refract_half4(half4 I, half4 N, half eta) { return __builtin_spirv_refract(I, N, eta); }


// CHECK-LABEL: define spir_func float @test_refract_float(
// CHECK-SAME: float noundef [[I:%.*]], float noundef [[N:%.*]], float noundef [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK:    [[SPV_REFRACT:%.*]] = tail call float @llvm.spv.refract.f32.f32(float [[I]], float [[N]], float [[ETA]])
// CHECK-NEXT:    ret float [[SPV_REFRACT]]
//
float test_refract_float(float I, float N, float eta) { return __builtin_spirv_refract(I, N, eta); }

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
