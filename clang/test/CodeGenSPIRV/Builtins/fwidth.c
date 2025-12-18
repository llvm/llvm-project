// RUN: %clang_cc1 -O1 -triple spirv-pc-vulkan-compute %s -emit-llvm -o - | FileCheck %s

typedef _Float16 half;
typedef half half2 __attribute__((ext_vector_type(2)));
typedef half half3 __attribute__((ext_vector_type(3)));
typedef half half4 __attribute__((ext_vector_type(4)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));

// CHECK: [[fwidth0:%.*]] = tail call half @llvm.spv.fwidth.f16(half {{%.*}})
// CHECK: ret half [[fwidth0]]
half test_fwidth_half(half X) { return __builtin_spirv_fwidth(X); }

// CHECK: [[fwidth0:%.*]] = tail call <2 x half> @llvm.spv.fwidth.v2f16(<2 x half>  {{%.*}})
// CHECK: ret <2 x half> [[fwidth0]]
half2 test_fwidth_half2(half2 X) { return __builtin_spirv_fwidth(X); }

// CHECK: [[fwidth0:%.*]] = tail call <3 x half> @llvm.spv.fwidth.v3f16(<3 x half> {{%.*}})
// CHECK: ret <3 x half> [[fwidth0]]
half3 test_fwidth_half3(half3 X) { return __builtin_spirv_fwidth(X); }

// CHECK: [[fwidth0:%.*]] = tail call <4 x half> @llvm.spv.fwidth.v4f16(<4 x half> {{%.*}})
// CHECK: ret <4 x half> [[fwidth0]]
half4 test_fwidth_half4(half4 X) { return __builtin_spirv_fwidth(X); }

// CHECK: [[fwidth0:%.*]] = tail call float @llvm.spv.fwidth.f32(float {{%.*}})
// CHECK: ret float [[fwidth0]]
float test_fwidth_float(float X) { return __builtin_spirv_fwidth(X); }

// CHECK: [[fwidth1:%.*]] = tail call <2 x float> @llvm.spv.fwidth.v2f32(<2 x float> {{%.*}})
// CHECK: ret <2 x float> [[fwidth1]]
float2 test_fwidth_float2(float2 X) { return __builtin_spirv_fwidth(X); }

// CHECK: [[fwidth2:%.*]] = tail call <3 x float> @llvm.spv.fwidth.v3f32(<3 x float> {{%.*}})
// CHECK: ret <3 x float> [[fwidth2]]
float3 test_fwidth_float3(float3 X) { return __builtin_spirv_fwidth(X); }

// CHECK: [[fwidth3:%.*]] = tail call <4 x float> @llvm.spv.fwidth.v4f32(<4 x float> {{%.*}})
// CHECK: ret <4 x float> [[fwidth3]]
float4 test_fwidth_float4(float4 X) { return __builtin_spirv_fwidth(X); }
