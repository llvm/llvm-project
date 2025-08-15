// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefix=SPVCHECK

// CHECK-LABEL: test_faceforward_half
// CHECK: %hlsl.dot.i = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, %{{.*}}
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt half %hlsl.dot.i, 0xH0000
// CHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn half %{{.*}}
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %cmp.i, half %{{.*}}, half %fneg.i
// CHECK: ret half %hlsl.select.i
// SPVCHECK-LABEL: test_faceforward_half
// SPVCHECK: %spv.faceforward.i = call reassoc nnan ninf nsz arcp afn noundef half @llvm.spv.faceforward.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
// SPVCHECK: ret half %spv.faceforward.i
half test_faceforward_half(half N, half I, half Ng) { return faceforward(N, I, Ng); }

// CHECK-LABEL: test_faceforward_half2
// CHECK: %hlsl.dot.i = call reassoc nnan ninf nsz arcp afn half @llvm.dx.fdot.v2f16(<2 x half> %{{.*}}, <2 x half> %{{.*}})
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt half %hlsl.dot.i, 0xH0000
// CHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <2 x half> %{{.*}}
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %cmp.i, <2 x half> %{{.*}}, <2 x half> %fneg.i
// CHECK: ret <2 x half> %hlsl.select.i
// SPVCHECK-LABEL: test_faceforward_half2
// SPVCHECK: %spv.faceforward.i = call reassoc nnan ninf nsz arcp afn noundef <2 x half> @llvm.spv.faceforward.v2f16(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
// SPVCHECK: ret <2 x half> %spv.faceforward.i
half2 test_faceforward_half2(half2 N, half2 I, half2 Ng) { return faceforward(N, I, Ng); }

// CHECK-LABEL: test_faceforward_half3
// CHECK: %hlsl.dot.i = call reassoc nnan ninf nsz arcp afn half @llvm.dx.fdot.v3f16(<3 x half> %{{.*}}, <3 x half> %{{.*}})
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt half %hlsl.dot.i, 0xH0000
// CHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <3 x half> %{{.*}}
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %cmp.i, <3 x half> %{{.*}}, <3 x half> %fneg.i
// CHECK: ret <3 x half> %hlsl.select.i
// SPVCHECK-LABEL: test_faceforward_half3
// SPVCHECK: %spv.faceforward.i = call reassoc nnan ninf nsz arcp afn noundef <3 x half> @llvm.spv.faceforward.v3f16(<3 x half> %{{.*}}, <3 x half> %{{.*}}, <3 x half> %{{.*}})
// SPVCHECK: ret <3 x half> %spv.faceforward.i
half3 test_faceforward_half3(half3 N, half3 I, half3 Ng) { return faceforward(N, I, Ng); }

// CHECK-LABEL: test_faceforward_half4
// CHECK: %hlsl.dot.i = call reassoc nnan ninf nsz arcp afn half @llvm.dx.fdot.v4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}})
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt half %hlsl.dot.i, 0xH0000
// CHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <4 x half> %{{.*}}
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %cmp.i, <4 x half> %{{.*}}, <4 x half> %fneg.i
// CHECK: ret <4 x half> %hlsl.select.i
// SPVCHECK-LABEL: test_faceforward_half4
// SPVCHECK: %spv.faceforward.i = call reassoc nnan ninf nsz arcp afn noundef <4 x half> @llvm.spv.faceforward.v4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <4 x half> %{{.*}})
// SPVCHECK: ret <4 x half> %spv.faceforward.i
half4 test_faceforward_half4(half4 N, half4 I, half4 Ng) { return faceforward(N, I, Ng); }

// CHECK-LABEL: test_faceforward_float
// CHECK: %hlsl.dot.i = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt float %hlsl.dot.i, 0.000000e+00
// CHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn float %{{.*}}
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %cmp.i, float %{{.*}}, float %fneg.i
// CHECK: ret float %hlsl.select.i
// SPVCHECK-LABEL: test_faceforward_float
// SPVCHECK: %spv.faceforward.i = call reassoc nnan ninf nsz arcp afn noundef float @llvm.spv.faceforward.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
// SPVCHECK: ret float %spv.faceforward.i
float test_faceforward_float(float N, float I, float Ng) { return faceforward(N, I, Ng); }

// CHECK-LABEL: test_faceforward_float2
// CHECK: %hlsl.dot.i = call reassoc nnan ninf nsz arcp afn float @llvm.dx.fdot.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt float %hlsl.dot.i, 0.000000e+00
// CHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %cmp.i, <2 x float> %{{.*}}, <2 x float> %fneg.i
// CHECK: ret <2 x float> %hlsl.select.i
// SPVCHECK-LABEL: test_faceforward_float2
// SPVCHECK: %spv.faceforward.i = call reassoc nnan ninf nsz arcp afn noundef <2 x float> @llvm.spv.faceforward.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// SPVCHECK: ret <2 x float> %spv.faceforward.i
float2 test_faceforward_float2(float2 N, float2 I, float2 Ng) { return faceforward(N, I, Ng); }

// CHECK-LABEL: test_faceforward_float3
// CHECK: %hlsl.dot.i = call reassoc nnan ninf nsz arcp afn float @llvm.dx.fdot.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}})
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt float %hlsl.dot.i, 0.000000e+00
// CHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %cmp.i, <3 x float> %{{.*}}, <3 x float> %fneg.i
// CHECK: ret <3 x float> %hlsl.select.i
// SPVCHECK-LABEL: test_faceforward_float3
// SPVCHECK: %spv.faceforward.i = call reassoc nnan ninf nsz arcp afn noundef <3 x float> @llvm.spv.faceforward.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}}, <3 x float> %{{.*}})
// SPVCHECK: ret <3 x float> %spv.faceforward.i
float3 test_faceforward_float3(float3 N, float3 I, float3 Ng) { return faceforward(N, I, Ng); }

// CHECK-LABEL: test_faceforward_float4
// CHECK: %hlsl.dot.i = call reassoc nnan ninf nsz arcp afn float @llvm.dx.fdot.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt float %hlsl.dot.i, 0.000000e+00
// CHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %cmp.i, <4 x float> %{{.*}}, <4 x float> %fneg.i
// CHECK: ret <4 x float> %hlsl.select.i
// SPVCHECK-LABEL: test_faceforward_float4
// SPVCHECK: %spv.faceforward.i = call reassoc nnan ninf nsz arcp afn noundef <4 x float> @llvm.spv.faceforward.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
// SPVCHECK: ret <4 x float> %spv.faceforward.i
float4 test_faceforward_float4(float4 N, float4 I, float4 Ng) { return faceforward(N, I, Ng); }
