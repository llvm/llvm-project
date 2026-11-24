// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,DXCHECK
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple spirv-unknown-vulkan-library %s -fnative-half-type -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,SPVCHECK

// CHECK-LABEL: test_smoothstep_half2x3
// DXCHECK: [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <6 x half> %{{.*}}, %{{.*}}
// DXCHECK: [[SUB1:%.*]] = fsub reassoc nnan ninf nsz arcp afn <6 x half> %{{.*}}, %{{.*}}
// DXCHECK: [[DIV:%.*]] = fdiv reassoc nnan ninf nsz arcp afn <6 x half> [[SUB]], [[SUB1]]
// DXCHECK: call reassoc nnan ninf nsz arcp afn <6 x half> @llvm.dx.saturate.v6f16(<6 x half> [[DIV]])
// DXCHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <6 x half> splat (half 2.000000e+00), %{{.*}}
// DXCHECK: [[SUB2:%.*]] = fsub reassoc nnan ninf nsz arcp afn <6 x half> splat (half 3.000000e+00), [[MUL]]
// DXCHECK: [[MUL1:%.*]] = fmul reassoc nnan ninf nsz arcp afn <6 x half> [[SUB2]], %{{.*}}
// DXCHECK: fmul reassoc nnan ninf nsz arcp afn <6 x half> [[MUL1]], %{{.*}}
// SPVCHECK: call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <6 x half> @llvm.spv.smoothstep.v6f16(<6 x half> %{{.*}}, <6 x half> %{{.*}}, <6 x half> %{{.*}})
half2x3 test_smoothstep_half2x3(half2x3 p0, half2x3 p1, half2x3 p2) { return smoothstep(p0, p1, p2); }

// CHECK-LABEL: test_smoothstep_half3x4
// DXCHECK: [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <12 x half> %{{.*}}, %{{.*}}
// DXCHECK: [[SUB1:%.*]] = fsub reassoc nnan ninf nsz arcp afn <12 x half> %{{.*}}, %{{.*}}
// DXCHECK: [[DIV:%.*]] = fdiv reassoc nnan ninf nsz arcp afn <12 x half> [[SUB]], [[SUB1]]
// DXCHECK: call reassoc nnan ninf nsz arcp afn <12 x half> @llvm.dx.saturate.v12f16(<12 x half> [[DIV]])
// DXCHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <12 x half> splat (half 2.000000e+00), %{{.*}}
// DXCHECK: [[SUB2:%.*]] = fsub reassoc nnan ninf nsz arcp afn <12 x half> splat (half 3.000000e+00), [[MUL]]
// DXCHECK: [[MUL1:%.*]] = fmul reassoc nnan ninf nsz arcp afn <12 x half> [[SUB2]], %{{.*}}
// DXCHECK: fmul reassoc nnan ninf nsz arcp afn <12 x half> [[MUL1]], %{{.*}}
// SPVCHECK: call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <12 x half> @llvm.spv.smoothstep.v12f16(<12 x half> %{{.*}}, <12 x half> %{{.*}}, <12 x half> %{{.*}})
half3x4 test_smoothstep_half3x4(half3x4 p0, half3x4 p1, half3x4 p2) { return smoothstep(p0, p1, p2); }

// CHECK-LABEL: test_smoothstep_half4x4
// DXCHECK: [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <16 x half> %{{.*}}, %{{.*}}
// DXCHECK: [[SUB1:%.*]] = fsub reassoc nnan ninf nsz arcp afn <16 x half> %{{.*}}, %{{.*}}
// DXCHECK: [[DIV:%.*]] = fdiv reassoc nnan ninf nsz arcp afn <16 x half> [[SUB]], [[SUB1]]
// DXCHECK: call reassoc nnan ninf nsz arcp afn <16 x half> @llvm.dx.saturate.v16f16(<16 x half> [[DIV]])
// DXCHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <16 x half> splat (half 2.000000e+00), %{{.*}}
// DXCHECK: [[SUB2:%.*]] = fsub reassoc nnan ninf nsz arcp afn <16 x half> splat (half 3.000000e+00), [[MUL]]
// DXCHECK: [[MUL1:%.*]] = fmul reassoc nnan ninf nsz arcp afn <16 x half> [[SUB2]], %{{.*}}
// DXCHECK: fmul reassoc nnan ninf nsz arcp afn <16 x half> [[MUL1]], %{{.*}}
// SPVCHECK: call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <16 x half> @llvm.spv.smoothstep.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
half4x4 test_smoothstep_half4x4(half4x4 p0, half4x4 p1, half4x4 p2) { return smoothstep(p0, p1, p2); }

// CHECK-LABEL: test_smoothstep_float2x3
// DXCHECK: [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <6 x float> %{{.*}}, %{{.*}}
// DXCHECK: [[SUB1:%.*]] = fsub reassoc nnan ninf nsz arcp afn <6 x float> %{{.*}}, %{{.*}}
// DXCHECK: [[DIV:%.*]] = fdiv reassoc nnan ninf nsz arcp afn <6 x float> [[SUB]], [[SUB1]]
// DXCHECK: call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.dx.saturate.v6f32(<6 x float> [[DIV]])
// DXCHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <6 x float> splat (float 2.000000e+00), %{{.*}}
// DXCHECK: [[SUB2:%.*]] = fsub reassoc nnan ninf nsz arcp afn <6 x float> splat (float 3.000000e+00), [[MUL]]
// DXCHECK: [[MUL1:%.*]] = fmul reassoc nnan ninf nsz arcp afn <6 x float> [[SUB2]], %{{.*}}
// DXCHECK: fmul reassoc nnan ninf nsz arcp afn <6 x float> [[MUL1]], %{{.*}}
// SPVCHECK: call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <6 x float> @llvm.spv.smoothstep.v6f32(<6 x float> %{{.*}}, <6 x float> %{{.*}}, <6 x float> %{{.*}})
float2x3 test_smoothstep_float2x3(float2x3 p0, float2x3 p1, float2x3 p2) { return smoothstep(p0, p1, p2); }

// CHECK-LABEL: test_smoothstep_float3x4
// DXCHECK: [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <12 x float> %{{.*}}, %{{.*}}
// DXCHECK: [[SUB1:%.*]] = fsub reassoc nnan ninf nsz arcp afn <12 x float> %{{.*}}, %{{.*}}
// DXCHECK: [[DIV:%.*]] = fdiv reassoc nnan ninf nsz arcp afn <12 x float> [[SUB]], [[SUB1]]
// DXCHECK: call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.dx.saturate.v12f32(<12 x float> [[DIV]])
// DXCHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <12 x float> splat (float 2.000000e+00), %{{.*}}
// DXCHECK: [[SUB2:%.*]] = fsub reassoc nnan ninf nsz arcp afn <12 x float> splat (float 3.000000e+00), [[MUL]]
// DXCHECK: [[MUL1:%.*]] = fmul reassoc nnan ninf nsz arcp afn <12 x float> [[SUB2]], %{{.*}}
// DXCHECK: fmul reassoc nnan ninf nsz arcp afn <12 x float> [[MUL1]], %{{.*}}
// SPVCHECK: call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <12 x float> @llvm.spv.smoothstep.v12f32(<12 x float> %{{.*}}, <12 x float> %{{.*}}, <12 x float> %{{.*}})
float3x4 test_smoothstep_float3x4(float3x4 p0, float3x4 p1, float3x4 p2) { return smoothstep(p0, p1, p2); }

// CHECK-LABEL: test_smoothstep_float4x4
// DXCHECK: [[SUB:%.*]] = fsub reassoc nnan ninf nsz arcp afn <16 x float> %{{.*}}, %{{.*}}
// DXCHECK: [[SUB1:%.*]] = fsub reassoc nnan ninf nsz arcp afn <16 x float> %{{.*}}, %{{.*}}
// DXCHECK: [[DIV:%.*]] = fdiv reassoc nnan ninf nsz arcp afn <16 x float> [[SUB]], [[SUB1]]
// DXCHECK: call reassoc nnan ninf nsz arcp afn <16 x float> @llvm.dx.saturate.v16f32(<16 x float> [[DIV]])
// DXCHECK: [[MUL:%.*]] = fmul reassoc nnan ninf nsz arcp afn <16 x float> splat (float 2.000000e+00), %{{.*}}
// DXCHECK: [[SUB2:%.*]] = fsub reassoc nnan ninf nsz arcp afn <16 x float> splat (float 3.000000e+00), [[MUL]]
// DXCHECK: [[MUL1:%.*]] = fmul reassoc nnan ninf nsz arcp afn <16 x float> [[SUB2]], %{{.*}}
// DXCHECK: fmul reassoc nnan ninf nsz arcp afn <16 x float> [[MUL1]], %{{.*}}
// SPVCHECK: call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <16 x float> @llvm.spv.smoothstep.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}})
float4x4 test_smoothstep_float4x4(float4x4 p0, float4x4 p1, float4x4 p2) { return smoothstep(p0, p1, p2); }
