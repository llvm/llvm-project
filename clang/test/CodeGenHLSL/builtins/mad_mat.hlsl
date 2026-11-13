// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,DXCHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,SPVCHECK,NO_HALF

using namespace hlsl;

// CHECK-LABEL: test_mad_half2x3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <6 x half> @llvm.fmuladd.v6f16(<6 x half> %{{.*}}, <6 x half> %{{.*}}, <6 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.fmuladd.v6f32(<6 x float> %{{.*}}, <6 x float> %{{.*}}, <6 x float> %{{.*}})
half2x3 test_mad_half2x3(half2x3 p0, half2x3 p1, half2x3 p2) { return mad(p0, p1, p2); }

// CHECK-LABEL: test_mad_half3x4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <12 x half> @llvm.fmuladd.v12f16(<12 x half> %{{.*}}, <12 x half> %{{.*}}, <12 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.fmuladd.v12f32(<12 x float> %{{.*}}, <12 x float> %{{.*}}, <12 x float> %{{.*}})
half3x4 test_mad_half3x4(half3x4 p0, half3x4 p1, half3x4 p2) { return mad(p0, p1, p2); }

// CHECK-LABEL: test_mad_half4x4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <16 x half> @llvm.fmuladd.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <16 x float> @llvm.fmuladd.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
half4x4 test_mad_half4x4(half4x4 p0, half4x4 p1, half4x4 p2) { return mad(p0, p1, p2); }

// CHECK-LABEL: test_mad_int2x3
// DXCHECK: call <6 x i32> @llvm.dx.imad.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}}, <6 x i32> %{{.*}})
// SPVCHECK: [[MUL:%.*]] = mul nsw <6 x i32> %{{.*}}, %{{.*}}
// SPVCHECK: add nsw <6 x i32> [[MUL]], %{{.*}}
int2x3 test_mad_int2x3(int2x3 p0, int2x3 p1, int2x3 p2) { return mad(p0, p1, p2); }

// CHECK-LABEL: test_mad_int3x4
// DXCHECK: call <12 x i32> @llvm.dx.imad.v12i32(<12 x i32> %{{.*}}, <12 x i32> %{{.*}}, <12 x i32> %{{.*}})
// SPVCHECK: [[MUL:%.*]] = mul nsw <12 x i32> %{{.*}}, %{{.*}}
// SPVCHECK: add nsw <12 x i32> [[MUL]], %{{.*}}
int3x4 test_mad_int3x4(int3x4 p0, int3x4 p1, int3x4 p2) { return mad(p0, p1, p2); }

// CHECK-LABEL: test_mad_int4x4
// DXCHECK: call <16 x i32> @llvm.dx.imad.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// SPVCHECK: [[MUL:%.*]] = mul nsw <16 x i32> %{{.*}}, %{{.*}}
// SPVCHECK: add nsw <16 x i32> [[MUL]], %{{.*}}
int4x4 test_mad_int4x4(int4x4 p0, int4x4 p1, int4x4 p2) { return mad(p0, p1, p2); }

// CHECK-LABEL: test_mad_uint2x3
// DXCHECK: call <6 x i32> @llvm.dx.umad.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}}, <6 x i32> %{{.*}})
// SPVCHECK: [[MUL:%.*]] = mul nuw <6 x i32> %{{.*}}, %{{.*}}
// SPVCHECK: add nuw <6 x i32> [[MUL]], %{{.*}}
uint2x3 test_mad_uint2x3(uint2x3 p0, uint2x3 p1, uint2x3 p2) { return mad(p0, p1, p2); }

// CHECK-LABEL: test_mad_uint3x4
// DXCHECK: call <12 x i32> @llvm.dx.umad.v12i32(<12 x i32> %{{.*}}, <12 x i32> %{{.*}}, <12 x i32> %{{.*}})
// SPVCHECK: [[MUL:%.*]] = mul nuw <12 x i32> %{{.*}}, %{{.*}}
// SPVCHECK: add nuw <12 x i32> [[MUL]], %{{.*}}
uint3x4 test_mad_uint3x4(uint3x4 p0, uint3x4 p1, uint3x4 p2) { return mad(p0, p1, p2); }

// CHECK-LABEL: test_mad_uint4x4
// DXCHECK: call <16 x i32> @llvm.dx.umad.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// SPVCHECK: [[MUL:%.*]] = mul nuw <16 x i32> %{{.*}}, %{{.*}}
// SPVCHECK: add nuw <16 x i32> [[MUL]], %{{.*}}
uint4x4 test_mad_uint4x4(uint4x4 p0, uint4x4 p1, uint4x4 p2) { return mad(p0, p1, p2); }
