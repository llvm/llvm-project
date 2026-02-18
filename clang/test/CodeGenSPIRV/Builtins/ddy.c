// RUN: %clang_cc1 -O1 -triple spirv-pc-vulkan-pixel %s -emit-llvm -o - | FileCheck %s

typedef _Float16 half;
typedef half half2 __attribute__((ext_vector_type(2)));
typedef half half3 __attribute__((ext_vector_type(3)));
typedef half half4 __attribute__((ext_vector_type(4)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));

// CHECK: [[ddy0:%.*]] = tail call half @llvm.spv.ddy.f16(half {{%.*}})
// CHECK: ret half [[ddy0]]
half test_ddy_half(half X) { return __builtin_spirv_ddy(X); }

// CHECK: [[ddy0:%.*]] = tail call <2 x half> @llvm.spv.ddy.v2f16(<2 x half>  {{%.*}})
// CHECK: ret <2 x half> [[ddy0]]
half2 test_ddy_half2(half2 X) { return __builtin_spirv_ddy(X); }

// CHECK: [[ddy0:%.*]] = tail call <3 x half> @llvm.spv.ddy.v3f16(<3 x half> {{%.*}})
// CHECK: ret <3 x half> [[ddy0]]
half3 test_ddy_half3(half3 X) { return __builtin_spirv_ddy(X); }

// CHECK: [[ddy0:%.*]] = tail call <4 x half> @llvm.spv.ddy.v4f16(<4 x half> {{%.*}})
// CHECK: ret <4 x half> [[ddy0]]
half4 test_ddy_half4(half4 X) { return __builtin_spirv_ddy(X); }

// CHECK: [[ddy0:%.*]] = tail call float @llvm.spv.ddy.f32(float {{%.*}})
// CHECK: ret float [[ddy0]]
float test_ddy_float(float X) { return __builtin_spirv_ddy(X); }

// CHECK: [[ddy1:%.*]] = tail call <2 x float> @llvm.spv.ddy.v2f32(<2 x float> {{%.*}})
// CHECK: ret <2 x float> [[ddy1]]
float2 test_ddy_float2(float2 X) { return __builtin_spirv_ddy(X); }

// CHECK: [[ddy2:%.*]] = tail call <3 x float> @llvm.spv.ddy.v3f32(<3 x float> {{%.*}})
// CHECK: ret <3 x float> [[ddy2]]
float3 test_ddy_float3(float3 X) { return __builtin_spirv_ddy(X); }

// CHECK: [[ddy3:%.*]] = tail call <4 x float> @llvm.spv.ddy.v4f32(<4 x float> {{%.*}})
// CHECK: ret <4 x float> [[ddy3]]
float4 test_ddy_float4(float4 X) { return __builtin_spirv_ddy(X); }
