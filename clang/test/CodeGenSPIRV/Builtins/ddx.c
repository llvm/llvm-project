// RUN: %clang_cc1 -O1 -triple spirv-pc-vulkan-pixel %s -emit-llvm -o - | FileCheck %s

typedef _Float16 half;
typedef half half2 __attribute__((ext_vector_type(2)));
typedef half half3 __attribute__((ext_vector_type(3)));
typedef half half4 __attribute__((ext_vector_type(4)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));

// CHECK: [[ddx0:%.*]] = tail call half @llvm.spv.ddx.f16(half {{%.*}})
// CHECK: ret half [[ddx0]]
half test_ddx_half(half X) { return __builtin_spirv_ddx(X); }

// CHECK: [[ddx0:%.*]] = tail call <2 x half> @llvm.spv.ddx.v2f16(<2 x half>  {{%.*}})
// CHECK: ret <2 x half> [[ddx0]]
half2 test_ddx_half2(half2 X) { return __builtin_spirv_ddx(X); }

// CHECK: [[ddx0:%.*]] = tail call <3 x half> @llvm.spv.ddx.v3f16(<3 x half> {{%.*}})
// CHECK: ret <3 x half> [[ddx0]]
half3 test_ddx_half3(half3 X) { return __builtin_spirv_ddx(X); }

// CHECK: [[ddx0:%.*]] = tail call <4 x half> @llvm.spv.ddx.v4f16(<4 x half> {{%.*}})
// CHECK: ret <4 x half> [[ddx0]]
half4 test_ddx_half4(half4 X) { return __builtin_spirv_ddx(X); }

// CHECK: [[ddx0:%.*]] = tail call float @llvm.spv.ddx.f32(float {{%.*}})
// CHECK: ret float [[ddx0]]
float test_ddx_float(float X) { return __builtin_spirv_ddx(X); }

// CHECK: [[ddx1:%.*]] = tail call <2 x float> @llvm.spv.ddx.v2f32(<2 x float> {{%.*}})
// CHECK: ret <2 x float> [[ddx1]]
float2 test_ddx_float2(float2 X) { return __builtin_spirv_ddx(X); }

// CHECK: [[ddx2:%.*]] = tail call <3 x float> @llvm.spv.ddx.v3f32(<3 x float> {{%.*}})
// CHECK: ret <3 x float> [[ddx2]]
float3 test_ddx_float3(float3 X) { return __builtin_spirv_ddx(X); }

// CHECK: [[ddx3:%.*]] = tail call <4 x float> @llvm.spv.ddx.v4f32(<4 x float> {{%.*}})
// CHECK: ret <4 x float> [[ddx3]]
float4 test_ddx_float4(float4 X) { return __builtin_spirv_ddx(X); }
