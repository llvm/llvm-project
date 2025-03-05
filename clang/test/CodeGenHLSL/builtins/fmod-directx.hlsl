// DirectX target:
//
// ---------- Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -O1 -o - | FileCheck %s \
// RUN:   -DFNATTRS="noundef nofpclass(nan inf)" -DTYPE=half
//
// ---------- No Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm \
// RUN:   -O1 -o - | FileCheck %s \
// RUN:   -DFNATTRS="noundef nofpclass(nan inf)" -DTYPE=float



// CHECK: define [[FNATTRS]] [[TYPE]] @
// CHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn [[TYPE]]
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge [[TYPE]]
// CHECK: %elt.abs.i = tail call reassoc nnan ninf nsz arcp afn [[TYPE]] @llvm.fabs.f
// CHECK: %hlsl.frac.i = tail call reassoc nnan ninf nsz arcp afn [[TYPE]] @llvm.dx.frac.f
// CHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn [[TYPE]]
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn [[TYPE]]
// CHECK: ret [[TYPE]] %mul.i
half test_fmod_half(half p0, half p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <2 x [[TYPE]]> @
// CHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]>
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <2 x [[TYPE]]>
// CHECK: %elt.abs.i = tail call reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]> @llvm.fabs.v2f
// CHECK: %hlsl.frac.i = tail call reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]> @llvm.dx.frac.v2f
// CHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]>
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <2 x i1>
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]>
// CHECK: ret <2 x [[TYPE]]> %mul.i
half2 test_fmod_half2(half2 p0, half2 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <3 x [[TYPE]]> @
// CHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]>
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <3 x [[TYPE]]>
// CHECK: %elt.abs.i = tail call reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]> @llvm.fabs.v3f
// CHECK: %hlsl.frac.i = tail call reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]> @llvm.dx.frac.v3f
// CHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]>
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <3 x i1>
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]>
// CHECK: ret <3 x [[TYPE]]> %mul.i
half3 test_fmod_half3(half3 p0, half3 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <4 x [[TYPE]]> @
// CHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]>
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <4 x [[TYPE]]>
// CHECK: %elt.abs.i = tail call reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]> @llvm.fabs.v4f
// CHECK: %hlsl.frac.i = tail call reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]> @llvm.dx.frac.v4f
// CHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]>
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <4 x i1>
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]>
// CHECK: ret <4 x [[TYPE]]> %mul.i
half4 test_fmod_half4(half4 p0, half4 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn float
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge float
// CHECK: %elt.abs.i = tail call reassoc nnan ninf nsz arcp afn float @llvm.fabs.f
// CHECK: %hlsl.frac.i = tail call reassoc nnan ninf nsz arcp afn float @llvm.dx.frac.f
// CHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn float
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn float 
// CHECK: ret float %mul.i
float test_fmod_float(float p0, float p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <2 x float>
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <2 x float>
// CHECK: %elt.abs.i = tail call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.fabs.v2f
// CHECK: %hlsl.frac.i = tail call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.dx.frac.v2f
// CHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn <2 x float>
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <2 x i1>
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <2 x float>
// CHECK: ret <2 x float> %mul.i
float2 test_fmod_float2(float2 p0, float2 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <3 x float>
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <3 x float>
// CHECK: %elt.abs.i = tail call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.fabs.v3f
// CHECK: %hlsl.frac.i = tail call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.dx.frac.v3f
// CHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn <3 x float>
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <3 x i1>
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <3 x float>
// CHECK: ret <3 x float> %mul.i
float3 test_fmod_float3(float3 p0, float3 p1) { return fmod(p0, p1); }

// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <4 x float>
// CHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <4 x float>
// CHECK: %elt.abs.i = tail call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.fabs.v4f
// CHECK: %hlsl.frac.i = tail call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.frac.v4f
// CHECK: %fneg2.i = fneg reassoc nnan ninf nsz arcp afn <4 x float>
// CHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <4 x i1>
// CHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <4 x float>
// CHECK: ret <4 x float> %mul.i
float4 test_fmod_float4(float4 p0, float4 p1) { return fmod(p0, p1); }

