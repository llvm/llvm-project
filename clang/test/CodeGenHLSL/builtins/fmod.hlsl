// DirectX target:
//
// ---------- Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,DXCHECK

//
// ---------- No Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,DXCHECK


// Spirv target:
//
// ---------- Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SPVCHECK

//
// ---------- No Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm \
// RUN:   -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SPVCHECK



// CHECK-LABEL: test_fmod_half
// DXCHECK: [[DIV:%.*]] = fdiv [[FMF:reassoc nnan ninf nsz arcp afn]] [[TYPE:half|float]] %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call [[FMF]] [[TYPE]] @llvm.fabs.[[INT_TYPE:f16|f32]]([[TYPE]] %{{.*}})
// DXCHECK: [[FRAC:%.*]] = call [[FMF]] [[TYPE]] @llvm.dx.frac.[[INT_TYPE]]([[TYPE]] [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp [[FMF]] oge [[TYPE]] %{{.*}}, 0
// DXCHECK: [[NEG:%.*]] = fneg [[FMF]] [[TYPE]] %{{.*}}
// DXCHECK: [[SELECT:%.*]] = select [[FMF]] i1 [[CMP]], [[TYPE]] %{{.*}}, [[TYPE]] [[NEG]]
// DXCHECK: [[RESULT:%.*]] = fmul [[FMF]] [[TYPE]] [[SELECT]], %{{.*}}
// SPVCHECK: [[RESULT:%.*]] = frem [[FMF:reassoc nnan ninf nsz arcp afn]] [[TYPE:half|float]] %{{.*}}, %{{.*}}
// CHECK: ret [[TYPE]] [[RESULT]]
half test_fmod_half(half p0, half p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_half2
// DXCHECK: [[DIV:%.*]] = fdiv [[FMF:reassoc nnan ninf nsz arcp afn]] <2 x [[TYPE:half|float]]> %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call [[FMF]] <2 x [[TYPE]]> @llvm.fabs.v2[[INT_TYPE:f16|f32]](<2 x [[TYPE]]> %{{.*}})
// DXCHECK: [[FRAC:%.*]] = call [[FMF]] <2 x [[TYPE]]> @llvm.dx.frac.v2[[INT_TYPE]](<2 x [[TYPE]]> [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp [[FMF]] oge <2 x [[TYPE]]> %{{.*}}, zeroinitializer
// DXCHECK: [[NEG:%.*]] = fneg [[FMF]] <2 x [[TYPE]]> %{{.*}}
// DXCHECK: [[SELECT:%.*]] = select [[FMF]] <2 x i1> [[CMP]], <2 x [[TYPE]]> %{{.*}}, <2 x [[TYPE]]> [[NEG]]
// DXCHECK: [[RESULT:%.*]] = fmul [[FMF]] <2 x [[TYPE]]> [[SELECT]], %{{.*}}
// SPVCHECK: [[RESULT:%.*]] = frem [[FMF:reassoc nnan ninf nsz arcp afn]] <2 x [[TYPE:half|float]]> %{{.*}}, %{{.*}}
// CHECK: ret <2 x [[TYPE]]> [[RESULT]]
half2 test_fmod_half2(half2 p0, half2 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_half3
// DXCHECK: [[DIV:%.*]] = fdiv [[FMF:reassoc nnan ninf nsz arcp afn]] <3 x [[TYPE:half|float]]> %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call [[FMF]] <3 x [[TYPE]]> @llvm.fabs.v3[[INT_TYPE:f16|f32]](<3 x [[TYPE]]> %{{.*}})
// DXCHECK: [[FRAC:%.*]] = call [[FMF]] <3 x [[TYPE]]> @llvm.dx.frac.v3[[INT_TYPE]](<3 x [[TYPE]]> [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp [[FMF]] oge <3 x [[TYPE]]> %{{.*}}, zeroinitializer
// DXCHECK: [[NEG:%.*]] = fneg [[FMF]] <3 x [[TYPE]]> %{{.*}}
// DXCHECK: [[SELECT:%.*]] = select [[FMF]] <3 x i1> [[CMP]], <3 x [[TYPE]]> %{{.*}}, <3 x [[TYPE]]> [[NEG]]
// DXCHECK: [[RESULT:%.*]] = fmul [[FMF]] <3 x [[TYPE]]> [[SELECT]], %{{.*}}
// SPVCHECK: [[RESULT:%.*]] = frem [[FMF:reassoc nnan ninf nsz arcp afn]] <3 x [[TYPE:half|float]]> %{{.*}}, %{{.*}}
// CHECK: ret <3 x [[TYPE]]> [[RESULT]]
half3 test_fmod_half3(half3 p0, half3 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_half4
// DXCHECK: [[DIV:%.*]] = fdiv [[FMF:reassoc nnan ninf nsz arcp afn]] <4 x [[TYPE:half|float]]> %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call [[FMF]] <4 x [[TYPE]]> @llvm.fabs.v4[[INT_TYPE:f16|f32]](<4 x [[TYPE]]> %{{.*}})
// DXCHECK: [[FRAC:%.*]] = call [[FMF]] <4 x [[TYPE]]> @llvm.dx.frac.v4[[INT_TYPE]](<4 x [[TYPE]]> [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp [[FMF]] oge <4 x [[TYPE]]> %{{.*}}, zeroinitializer
// DXCHECK: [[NEG:%.*]] = fneg [[FMF]] <4 x [[TYPE]]> %{{.*}}
// DXCHECK: [[SELECT:%.*]] = select [[FMF]] <4 x i1> [[CMP]], <4 x [[TYPE]]> %{{.*}}, <4 x [[TYPE]]> [[NEG]]
// DXCHECK: [[RESULT:%.*]] = fmul [[FMF]] <4 x [[TYPE]]> [[SELECT]], %{{.*}}
// SPVCHECK: [[RESULT:%.*]] = frem [[FMF:reassoc nnan ninf nsz arcp afn]] <4 x [[TYPE:half|float]]> %{{.*}}, %{{.*}}
// CHECK: ret <4 x [[TYPE]]> [[RESULT]]
half4 test_fmod_half4(half4 p0, half4 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_float
// DXCHECK: [[DIV:%.*]] = fdiv [[FMF:reassoc nnan ninf nsz arcp afn]] float %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call [[FMF]] float @llvm.fabs.f32(float %{{.*}})
// DXCHECK: [[FRAC:%.*]] = call [[FMF]] float @llvm.dx.frac.f32(float [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp [[FMF]] oge float %{{.*}}, 0.000000e+00
// DXCHECK: [[NEG:%.*]] = fneg [[FMF]] float %{{.*}}
// DXCHECK: [[SELECT:%.*]] = select [[FMF]] i1 [[CMP]], float %{{.*}}, float [[NEG]]
// DXCHECK: [[RESULT:%.*]] = fmul [[FMF]] float [[SELECT]], %{{.*}}
// SPVCHECK: [[RESULT:%.*]] = frem [[FMF:reassoc nnan ninf nsz arcp afn]] float %{{.*}}, %{{.*}}
// CHECK: ret float [[RESULT]]
float test_fmod_float(float p0, float p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_float2
// DXCHECK: [[DIV:%.*]] = fdiv [[FMF:reassoc nnan ninf nsz arcp afn]] <2 x float> %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call [[FMF]] <2 x float> @llvm.fabs.v2f32(<2 x float> %{{.*}})
// DXCHECK: [[FRAC:%.*]] = call [[FMF]] <2 x float> @llvm.dx.frac.v2f32(<2 x float> [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp [[FMF]] oge <2 x float> %{{.*}}, zeroinitializer
// DXCHECK: [[NEG:%.*]] = fneg [[FMF]] <2 x float> %{{.*}}
// DXCHECK: [[SELECT:%.*]] = select [[FMF]] <2 x i1> [[CMP]], <2 x float> %{{.*}}, <2 x float> [[NEG]]
// DXCHECK: [[RESULT:%.*]] = fmul [[FMF]] <2 x float> [[SELECT]], %{{.*}}
// SPVCHECK: [[RESULT:%.*]] = frem [[FMF:reassoc nnan ninf nsz arcp afn]] <2 x float> %{{.*}}, %{{.*}}
// CHECK: ret <2 x float> [[RESULT]]
float2 test_fmod_float2(float2 p0, float2 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_float3
// DXCHECK: [[DIV:%.*]] = fdiv [[FMF:reassoc nnan ninf nsz arcp afn]] <3 x float> %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call [[FMF]] <3 x float> @llvm.fabs.v3f32(<3 x float> %{{.*}})
// DXCHECK: [[FRAC:%.*]] = call [[FMF]] <3 x float> @llvm.dx.frac.v3f32(<3 x float> [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp [[FMF]] oge <3 x float> %{{.*}}, zeroinitializer
// DXCHECK: [[NEG:%.*]] = fneg [[FMF]] <3 x float> %{{.*}}
// DXCHECK: [[SELECT:%.*]] = select [[FMF]] <3 x i1> [[CMP]], <3 x float> %{{.*}}, <3 x float> [[NEG]]
// DXCHECK: [[RESULT:%.*]] = fmul [[FMF]] <3 x float> [[SELECT]], %{{.*}}
// SPVCHECK: [[RESULT:%.*]] = frem [[FMF:reassoc nnan ninf nsz arcp afn]] <3 x float> %{{.*}}, %{{.*}}
// CHECK: ret <3 x float> [[RESULT]]
float3 test_fmod_float3(float3 p0, float3 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_float4
// DXCHECK: [[DIV:%.*]] = fdiv [[FMF:reassoc nnan ninf nsz arcp afn]] <4 x float> %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call [[FMF]] <4 x float> @llvm.fabs.v4f32(<4 x float> %{{.*}})
// DXCHECK: [[FRAC:%.*]] = call [[FMF]] <4 x float> @llvm.dx.frac.v4f32(<4 x float> [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp [[FMF]] oge <4 x float> %{{.*}}, zeroinitializer
// DXCHECK: [[NEG:%.*]] = fneg [[FMF]] <4 x float> %{{.*}}
// DXCHECK: [[SELECT:%.*]] = select [[FMF]] <4 x i1> [[CMP]], <4 x float> %{{.*}}, <4 x float> [[NEG]]
// DXCHECK: [[RESULT:%.*]] = fmul [[FMF]] <4 x float> [[SELECT]], %{{.*}}
// SPVCHECK: [[RESULT:%.*]] = frem [[FMF:reassoc nnan ninf nsz arcp afn]] <4 x float> %{{.*}}, %{{.*}}
// CHECK: ret <4 x float> [[RESULT]]
float4 test_fmod_float4(float4 p0, float4 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_float5
// DXCHECK: [[DIV:%.*]] = fdiv [[FMF:reassoc nnan ninf nsz arcp afn]] <5 x float> %{{.*}}, %{{.*}}
// DXCHECK: [[ABS:%.*]] = call [[FMF]] <5 x float> @llvm.fabs.v5f32(<5 x float> %{{.*}})
// DXCHECK: [[FRAC:%.*]] = call [[FMF]] <5 x float> @llvm.dx.frac.v5f32(<5 x float> [[ABS]])
// DXCHECK: [[CMP:%.*]] = fcmp [[FMF]] oge <5 x float> %{{.*}}, zeroinitializer
// DXCHECK: [[NEG:%.*]] = fneg [[FMF]] <5 x float> %{{.*}}
// DXCHECK: [[SELECT:%.*]] = select [[FMF]] <5 x i1> [[CMP]], <5 x float> %{{.*}}, <5 x float> [[NEG]]
// DXCHECK: [[RESULT:%.*]] = fmul [[FMF]] <5 x float> [[SELECT]], %{{.*}}
// SPVCHECK: [[RESULT:%.*]] = frem [[FMF:reassoc nnan ninf nsz arcp afn]] <5 x float> %{{.*}}, %{{.*}}
// CHECK: ret <5 x float> [[RESULT]]
vector<float, 5> test_fmod_float5(vector<float, 5> p0,
                                  vector<float, 5> p1) {
  return fmod(p0, p1);
}

