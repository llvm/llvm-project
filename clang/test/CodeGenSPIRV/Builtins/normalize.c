// RUN: %clang_cc1 -O1 -triple spirv-pc-vulkan-compute %s -emit-llvm -o - | FileCheck %s

typedef _Float16 half;
typedef half half2 __attribute__((ext_vector_type(2)));
typedef half half3 __attribute__((ext_vector_type(3)));
typedef half half4 __attribute__((ext_vector_type(4)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));

// CHECK: [[NORM:%.*]] = tail call half @llvm.spv.normalize.f16(half {{%.*}})
// CHECK: ret half [[NORM]]
half test_normalize_half(half X) { return __builtin_spirv_normalize(X); }

// CHECK: [[NORM:%.*]] = tail call <2 x half> @llvm.spv.normalize.v2f16(<2 x half> {{%.*}})
// CHECK: ret <2 x half> [[NORM]]
half2 test_normalize_half2(half2 X) { return __builtin_spirv_normalize(X); }

// CHECK: [[NORM:%.*]] = tail call <3 x half> @llvm.spv.normalize.v3f16(<3 x half> {{%.*}})
// CHECK: ret <3 x half> [[NORM]]
half3 test_normalize_half3(half3 X) { return __builtin_spirv_normalize(X); }

// CHECK: [[NORM:%.*]] = tail call <4 x half> @llvm.spv.normalize.v4f16(<4 x half> {{%.*}})
// CHECK: ret <4 x half> [[NORM]]
half4 test_normalize_half4(half4 X) { return __builtin_spirv_normalize(X); }

// CHECK: [[NORM:%.*]] = tail call float @llvm.spv.normalize.f32(float {{%.*}})
// CHECK: ret float [[NORM]]
float test_normalize_float(float X) { return __builtin_spirv_normalize(X); }

// CHECK: [[NORM:%.*]] = tail call <2 x float> @llvm.spv.normalize.v2f32(<2 x float> {{%.*}})
// CHECK: ret <2 x float> [[NORM]]
float2 test_normalize_float2(float2 X) { return __builtin_spirv_normalize(X); }

// CHECK: [[NORM:%.*]] = tail call <3 x float> @llvm.spv.normalize.v3f32(<3 x float> {{%.*}})
// CHECK: ret <3 x float> [[NORM]]
float3 test_normalize_float3(float3 X) { return __builtin_spirv_normalize(X); }

// CHECK: [[NORM:%.*]] = tail call <4 x float> @llvm.spv.normalize.v4f32(<4 x float> {{%.*}})
// CHECK: ret <4 x float> [[NORM]]
float4 test_normalize_float4(float4 X) { return __builtin_spirv_normalize(X); }
