// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,DXCHECK
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple spirv-unknown-vulkan-library %s -fnative-half-type -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,SPVCHECK

// CHECK-LABEL: test_fmod_half2x3
// DXCHECK: fdiv reassoc nnan ninf nsz arcp afn <6 x half> %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call reassoc nnan ninf nsz arcp afn <6 x half> @llvm.fabs.v6f16(<6 x half> %{{.*}})
// DXCHECK: call reassoc nnan ninf nsz arcp afn <6 x half> @llvm.dx.frac.v6f16(<6 x half> [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn oge <6 x half> %{{.*}}, zeroinitializer
// DXCHECK: [[NEG:%.*]] = fneg reassoc nnan ninf nsz arcp afn <6 x half> %{{.*}}
// DXCHECK: [[SEL:%.*]] = select reassoc nnan ninf nsz arcp afn <6 x i1> [[CMP]], <6 x half> %{{.*}}, <6 x half> [[NEG]]
// DXCHECK: fmul reassoc nnan ninf nsz arcp afn <6 x half> [[SEL]], %{{.*}}
// SPVCHECK: frem reassoc nnan ninf nsz arcp afn <6 x half> %{{.*}}, %{{.*}}
half2x3 test_fmod_half2x3(half2x3 p0, half2x3 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_half3x4
// DXCHECK: fdiv reassoc nnan ninf nsz arcp afn <12 x half> %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call reassoc nnan ninf nsz arcp afn <12 x half> @llvm.fabs.v12f16(<12 x half> %{{.*}})
// DXCHECK: call reassoc nnan ninf nsz arcp afn <12 x half> @llvm.dx.frac.v12f16(<12 x half> [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn oge <12 x half> %{{.*}}, zeroinitializer
// DXCHECK: [[NEG:%.*]] = fneg reassoc nnan ninf nsz arcp afn <12 x half> %{{.*}}
// DXCHECK: [[SEL:%.*]] = select reassoc nnan ninf nsz arcp afn <12 x i1> [[CMP]], <12 x half> %{{.*}}, <12 x half> [[NEG]]
// DXCHECK: fmul reassoc nnan ninf nsz arcp afn <12 x half> [[SEL]], %{{.*}}
// SPVCHECK: frem reassoc nnan ninf nsz arcp afn <12 x half> %{{.*}}, %{{.*}}
half3x4 test_fmod_half3x4(half3x4 p0, half3x4 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_half4x4
// DXCHECK: fdiv reassoc nnan ninf nsz arcp afn <16 x half> %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call reassoc nnan ninf nsz arcp afn <16 x half> @llvm.fabs.v16f16(<16 x half> %{{.*}})
// DXCHECK: call reassoc nnan ninf nsz arcp afn <16 x half> @llvm.dx.frac.v16f16(<16 x half> [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn oge <16 x half> %{{.*}}, zeroinitializer
// DXCHECK: [[NEG:%.*]] = fneg reassoc nnan ninf nsz arcp afn <16 x half> %{{.*}}
// DXCHECK: [[SEL:%.*]] = select reassoc nnan ninf nsz arcp afn <16 x i1> [[CMP]], <16 x half> %{{.*}}, <16 x half> [[NEG]]
// DXCHECK: fmul reassoc nnan ninf nsz arcp afn <16 x half> [[SEL]], %{{.*}}
// SPVCHECK: frem reassoc nnan ninf nsz arcp afn <16 x half> %{{.*}}, %{{.*}}
half4x4 test_fmod_half4x4(half4x4 p0, half4x4 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_float2x3
// DXCHECK: fdiv reassoc nnan ninf nsz arcp afn <6 x float> %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.fabs.v6f32(<6 x float> %{{.*}})
// DXCHECK: call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.dx.frac.v6f32(<6 x float> [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn oge <6 x float> %{{.*}}, zeroinitializer
// DXCHECK: [[NEG:%.*]] = fneg reassoc nnan ninf nsz arcp afn <6 x float> %{{.*}}
// DXCHECK: [[SEL:%.*]] = select reassoc nnan ninf nsz arcp afn <6 x i1> [[CMP]], <6 x float> %{{.*}}, <6 x float> [[NEG]]
// DXCHECK: fmul reassoc nnan ninf nsz arcp afn <6 x float> [[SEL]], %{{.*}}
// SPVCHECK: frem reassoc nnan ninf nsz arcp afn <6 x float> %{{.*}}, %{{.*}}
float2x3 test_fmod_float2x3(float2x3 p0, float2x3 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_float3x4
// DXCHECK: fdiv reassoc nnan ninf nsz arcp afn <12 x float> %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.fabs.v12f32(<12 x float> %{{.*}})
// DXCHECK: call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.dx.frac.v12f32(<12 x float> [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn oge <12 x float> %{{.*}}, zeroinitializer
// DXCHECK: [[NEG:%.*]] = fneg reassoc nnan ninf nsz arcp afn <12 x float> %{{.*}}
// DXCHECK: [[SEL:%.*]] = select reassoc nnan ninf nsz arcp afn <12 x i1> [[CMP]], <12 x float> %{{.*}}, <12 x float> [[NEG]]
// DXCHECK: fmul reassoc nnan ninf nsz arcp afn <12 x float> [[SEL]], %{{.*}}
// SPVCHECK: frem reassoc nnan ninf nsz arcp afn <12 x float> %{{.*}}, %{{.*}}
float3x4 test_fmod_float3x4(float3x4 p0, float3x4 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_float4x4
// DXCHECK: fdiv reassoc nnan ninf nsz arcp afn <16 x float> %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call reassoc nnan ninf nsz arcp afn <16 x float> @llvm.fabs.v16f32(<16 x float> %{{.*}})
// DXCHECK: call reassoc nnan ninf nsz arcp afn <16 x float> @llvm.dx.frac.v16f32(<16 x float> [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn oge <16 x float> %{{.*}}, zeroinitializer
// DXCHECK: [[NEG:%.*]] = fneg reassoc nnan ninf nsz arcp afn <16 x float> %{{.*}}
// DXCHECK: [[SEL:%.*]] = select reassoc nnan ninf nsz arcp afn <16 x i1> [[CMP]], <16 x float> %{{.*}}, <16 x float> [[NEG]]
// DXCHECK: fmul reassoc nnan ninf nsz arcp afn <16 x float> [[SEL]], %{{.*}}
// SPVCHECK: frem reassoc nnan ninf nsz arcp afn <16 x float> %{{.*}}, %{{.*}}
float4x4 test_fmod_float4x4(float4x4 p0, float4x4 p1) { return fmod(p0, p1); }
