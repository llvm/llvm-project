// DirectX target:
//
// ---------- Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -o - | FileCheck %s \
// RUN:   -DTYPE=half --check-prefixes=CHECK,DXCHECK,DXNATIVE_HALF

//
// ---------- No Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm \
// RUN:   -o - | FileCheck %s \
// RUN:   -DTYPE=float --check-prefixes=CHECK,DXCHECK,DXNO_HALF


// Spirv target:
//
// ---------- Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -o - | FileCheck %s \
// RUN:   -DTYPE=half --check-prefixes=CHECK,SPVCHECK

//
// ---------- No Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm \
// RUN:   -o - | FileCheck %s \
// RUN:   -DTYPE=float --check-prefixes=CHECK,SPVCHECK



// CHECK-LABEL: test_fmod_half
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn [[TYPE]]
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn [[TYPE]]
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge [[TYPE]]
// DXNATIVE_HALF: %elt.abs.i = call reassoc nnan ninf nsz arcp afn [[TYPE]] @llvm.fabs.f16([[TYPE]]
// DXNO_HALF: %elt.abs.i = call reassoc nnan ninf nsz arcp afn [[TYPE]] @llvm.fabs.f32([[TYPE]]
// DXNATIVE_HALF: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn [[TYPE]] @llvm.dx.frac.f16([[TYPE]]
// DXNO_HALF: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn [[TYPE]] @llvm.dx.frac.f32([[TYPE]]
// DXCHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn [[TYPE]]
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn [[TYPE]]
// DXCHECK: ret [[TYPE]] %mul.i
// SPVCHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn [[TYPE]]
// SPVCHECK: ret [[TYPE]] %fmod.i
half test_fmod_half(half p0, half p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_half2
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]>
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]>
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <2 x [[TYPE]]>
// DXNATIVE_HALF: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]> @llvm.fabs.v2f16(<2 x [[TYPE]]>
// DXNO_HALF: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]> @llvm.fabs.v2f32(<2 x [[TYPE]]>
// DXNATIVE_HALF: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]> @llvm.dx.frac.v2f16(<2 x [[TYPE]]>
// DXNO_HALF: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]> @llvm.dx.frac.v2f32(<2 x [[TYPE]]>
// DXCHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]>
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <2 x i1>
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]>
// DXCHECK: ret <2 x [[TYPE]]> %mul.i
// SPVCHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]>
// SPVCHECK: ret <2 x [[TYPE]]> %fmod.i
half2 test_fmod_half2(half2 p0, half2 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_half3
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]>
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]>
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <3 x [[TYPE]]>
// DXNATIVE_HALF: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]> @llvm.fabs.v3f16(<3 x [[TYPE]]>
// DXNO_HALF: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]> @llvm.fabs.v3f32(<3 x [[TYPE]]>
// DXNATIVE_HALF: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]> @llvm.dx.frac.v3f16(<3 x [[TYPE]]>
// DXNO_HALF: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]> @llvm.dx.frac.v3f32(<3 x [[TYPE]]>
// DXCHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]>
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <3 x i1>
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]>
// DXCHECK: ret <3 x [[TYPE]]> %mul.i
// SPVCHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]>
// SPVCHECK: ret <3 x [[TYPE]]> %fmod.i
half3 test_fmod_half3(half3 p0, half3 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_half4
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]>
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]>
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <4 x [[TYPE]]>
// DXNATIVE_HALF: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]> @llvm.fabs.v4f16(<4 x [[TYPE]]>
// DXNO_HALF: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]> @llvm.fabs.v4f32(<4 x [[TYPE]]>
// DXNATIVE_HALF: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]> @llvm.dx.frac.v4f16(<4 x [[TYPE]]>
// DXNO_HALF: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]> @llvm.dx.frac.v4f32(<4 x [[TYPE]]>
// DXCHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]>
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <4 x i1>
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]>
// DXCHECK: ret <4 x [[TYPE]]> %mul.i
// SPVCHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]>
// SPVCHECK: ret <4 x [[TYPE]]> %fmod.i
half4 test_fmod_half4(half4 p0, half4 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_float
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn float
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn float
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge float
// DXCHECK: %elt.abs.i = call reassoc nnan ninf nsz arcp afn float @llvm.fabs.f32(float
// DXCHECK: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn float @llvm.dx.frac.f32(float
// DXCHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn float
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn float 
// DXCHECK: ret float %mul.i
// SPVCHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn float
// SPVCHECK: ret float %fmod.i
float test_fmod_float(float p0, float p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_float2
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <2 x float>
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <2 x float>
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <2 x float>
// DXCHECK: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.fabs.v2f32(<2 x float>
// DXCHECK: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.dx.frac.v2f32(<2 x float>
// DXCHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn <2 x float>
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <2 x i1>
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <2 x float>
// DXCHECK: ret <2 x float> %mul.i
// SPVCHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <2 x float>
// SPVCHECK: ret <2 x float> %fmod.i
float2 test_fmod_float2(float2 p0, float2 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_float3
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <3 x float>
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <3 x float>
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <3 x float>
// DXCHECK: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.fabs.v3f32(<3 x float>
// DXCHECK: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.dx.frac.v3f32(<3 x float>
// DXCHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn <3 x float>
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <3 x i1>
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <3 x float>
// DXCHECK: ret <3 x float> %mul.i
// SPVCHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <3 x float>
// SPVCHECK: ret <3 x float> %fmod.i
float3 test_fmod_float3(float3 p0, float3 p1) { return fmod(p0, p1); }

// CHECK-LABEL: test_fmod_float4
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <4 x float>
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <4 x float>
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <4 x float>
// DXCHECK: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.fabs.v4f32(<4 x float>
// DXCHECK: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.frac.v4f32(<4 x float>
// DXCHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn <4 x float>
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <4 x i1>
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <4 x float>
// DXCHECK: ret <4 x float> %mul.i
// SPVCHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <4 x float>
// SPVCHECK: ret <4 x float> %fmod.i
float4 test_fmod_float4(float4 p0, float4 p1) { return fmod(p0, p1); }

