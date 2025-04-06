// DirectX target:
//
// ---------- Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -o - | FileCheck %s -DFNATTRS="noundef nofpclass(nan inf)" \
// RUN:   -DTYPE=half -DINT_TYPE=f16 --check-prefixes=DXCHECK

//
// ---------- No Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm \
// RUN:   -o - | FileCheck %s -DFNATTRS="noundef nofpclass(nan inf)" \
// RUN:   -DTYPE=float -DINT_TYPE=f32 --check-prefixes=DXCHECK


// Spirv target:
//
// ---------- Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -o - | FileCheck %s \
// RUN:   -DFNATTRS="spir_func noundef nofpclass(nan inf)" -DTYPE=half

//
// ---------- No Native Half support test -----------
//
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm \
// RUN:   -o - | FileCheck %s \
// RUN:   -DFNATTRS="spir_func noundef nofpclass(nan inf)" -DTYPE=float



// DXCHECK: define [[FNATTRS]] [[TYPE]] @
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn [[TYPE]] %{{.*}}, %{{.*}}
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge [[TYPE]] %{{.*}}, 0
// DXCHECK: %elt.abs.i = call reassoc nnan ninf nsz arcp afn [[TYPE]] @llvm.fabs.[[INT_TYPE]]([[TYPE]] %{{.*}})
// DXCHECK: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn [[TYPE]] @llvm.dx.frac.[[INT_TYPE]]([[TYPE]] %elt.abs.i)
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn [[TYPE]] %{{.*}}
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, [[TYPE]] %{{.*}}, [[TYPE]] %fneg.i
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn [[TYPE]] %hlsl.select.i, %{{.*}}
// DXCHECK: ret [[TYPE]] %mul.i
// CHECK: define [[FNATTRS]] [[TYPE]] @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn [[TYPE]]
// CHECK: ret [[TYPE]] %fmod.i
half test_fmod_half(half p0, half p1) { return fmod(p0, p1); }

// DXCHECK: define [[FNATTRS]] <2 x [[TYPE]]> @
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]> %{{.*}}, %{{.*}}
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <2 x [[TYPE]]> %{{.*}}, zeroinitializer
// DXCHECK: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]> @llvm.fabs.v2[[INT_TYPE]](<2 x [[TYPE]]> %{{.*}})
// DXCHECK: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]> @llvm.dx.frac.v2[[INT_TYPE]](<2 x [[TYPE]]> %elt.abs.i)
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]> %{{.*}}
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <2 x i1> %{{.*}}, <2 x [[TYPE]]> %{{.*}}, <2 x [[TYPE]]> %fneg.i
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]> %hlsl.select.i, %{{.*}}
// DXCHECK: ret <2 x [[TYPE]]> %mul.i
// CHECK: define [[FNATTRS]] <2 x [[TYPE]]> @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <2 x [[TYPE]]>
// CHECK: ret <2 x [[TYPE]]> %fmod.i
half2 test_fmod_half2(half2 p0, half2 p1) { return fmod(p0, p1); }

// DXCHECK: define [[FNATTRS]] <3 x [[TYPE]]> @
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]> %{{.*}}, %{{.*}}
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <3 x [[TYPE]]> %{{.*}}, zeroinitializer
// DXCHECK: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]> @llvm.fabs.v3[[INT_TYPE]](<3 x [[TYPE]]> %{{.*}})
// DXCHECK: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]> @llvm.dx.frac.v3[[INT_TYPE]](<3 x [[TYPE]]> %elt.abs.i)
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]> %{{.*}}
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <3 x i1> %{{.*}}, <3 x [[TYPE]]> %{{.*}}, <3 x [[TYPE]]> %fneg.i
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]> %hlsl.select.i, %{{.*}}
// DXCHECK: ret <3 x [[TYPE]]> %mul.i
// CHECK: define [[FNATTRS]] <3 x [[TYPE]]> @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <3 x [[TYPE]]>
// CHECK: ret <3 x [[TYPE]]> %fmod.i
half3 test_fmod_half3(half3 p0, half3 p1) { return fmod(p0, p1); }

// DXCHECK: define [[FNATTRS]] <4 x [[TYPE]]> @
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]> %{{.*}}, %{{.*}}
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <4 x [[TYPE]]> %{{.*}}, zeroinitializer
// DXCHECK: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]> @llvm.fabs.v4[[INT_TYPE]](<4 x [[TYPE]]> %{{.*}})
// DXCHECK: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]> @llvm.dx.frac.v4[[INT_TYPE]](<4 x [[TYPE]]> %elt.abs.i)
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]> %{{.*}}
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <4 x i1> %{{.*}}, <4 x [[TYPE]]> %{{.*}}, <4 x [[TYPE]]> %fneg.i
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]> %hlsl.select.i, %{{.*}}
// DXCHECK: ret <4 x [[TYPE]]> %mul.i
// CHECK: define [[FNATTRS]] <4 x [[TYPE]]> @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <4 x [[TYPE]]>
// CHECK: ret <4 x [[TYPE]]> %fmod.i
half4 test_fmod_half4(half4 p0, half4 p1) { return fmod(p0, p1); }

// DXCHECK: define [[FNATTRS]] float @
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge float %{{.*}}, 0.000000e+00
// DXCHECK: %elt.abs.i = call reassoc nnan ninf nsz arcp afn float @llvm.fabs.f32(float %{{.*}})
// DXCHECK: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn float @llvm.dx.frac.f32(float %elt.abs.i)
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn float %{{.*}}
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %{{.*}}, float %{{.*}}, float %fneg.i
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn float %hlsl.select.i, %{{.*}}
// DXCHECK: ret float %mul.i
// CHECK: define [[FNATTRS]] float @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn float
// CHECK: ret float %fmod.i
float test_fmod_float(float p0, float p1) { return fmod(p0, p1); }

// DXCHECK: define [[FNATTRS]] <2 x float> @
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, %{{.*}}
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <2 x float> %{{.*}}, zeroinitializer
// DXCHECK: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.fabs.v2f32(<2 x float> %{{.*}})
// DXCHECK: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.dx.frac.v2f32(<2 x float> %elt.abs.i)
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <2 x i1> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %fneg.i
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <2 x float> %hlsl.select.i, %{{.*}}
// DXCHECK: ret <2 x float> %mul.i
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <2 x float>
// CHECK: ret <2 x float> %fmod.i
float2 test_fmod_float2(float2 p0, float2 p1) { return fmod(p0, p1); }

// DXCHECK: define [[FNATTRS]] <3 x float> @
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}, %{{.*}}
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <3 x float> %{{.*}}, zeroinitializer
// DXCHECK: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.fabs.v3f32(<3 x float> %{{.*}})
// DXCHECK: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.dx.frac.v3f32(<3 x float> %elt.abs.i)
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <3 x float> %{{.*}}
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <3 x i1> %{{.*}}, <3 x float> %{{.*}}, <3 x float> %fneg.i
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <3 x float> %hlsl.select.i, %{{.*}}
// DXCHECK: ret <3 x float> %mul.i
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <3 x float>
// CHECK: ret <3 x float> %fmod.i
float3 test_fmod_float3(float3 p0, float3 p1) { return fmod(p0, p1); }

// DXCHECK: define [[FNATTRS]] <4 x float> @
// DXCHECK: %div1.i = fdiv reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// DXCHECK: %cmp.i = fcmp reassoc nnan ninf nsz arcp afn oge <4 x float> %{{.*}}, zeroinitializer
// DXCHECK: %elt.abs.i = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.fabs.v4f32(<4 x float> %{{.*}})
// DXCHECK: %hlsl.frac.i = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.frac.v4f32(<4 x float> %elt.abs.i)
// DXCHECK: %fneg.i = fneg reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}
// DXCHECK: %hlsl.select.i = select reassoc nnan ninf nsz arcp afn <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %fneg.i
// DXCHECK: %mul.i = fmul reassoc nnan ninf nsz arcp afn <4 x float> %hlsl.select.i, %{{.*}}
// DXCHECK: ret <4 x float> %mul.i
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %fmod.i = frem reassoc nnan ninf nsz arcp afn <4 x float>
// CHECK: ret <4 x float> %fmod.i
float4 test_fmod_float4(float4 p0, float4 p1) { return fmod(p0, p1); }

