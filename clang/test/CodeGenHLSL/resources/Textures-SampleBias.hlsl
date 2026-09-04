// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-TEXEL,CHECK-OFFSET,DXIL-OFFSET \
// RUN:   -DDXIL_TY=2 -DRW=0 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCube -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCube -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-NOTEXEL,CHECK-NOOFFSET,DXIL-NOOFFSET \
// RUN:   -DDXIL_TY=5 -DRW=0 -DDIM=3
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCubeArray -DCOORD_DIM=4 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-NOTEXEL,CHECK-NOOFFSET,DXIL-NOOFFSET \
// RUN:   -DDXIL_TY=9 -DRW=0 -DDIM=3
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-TEXEL,CHECK-OFFSET,SPIRV-OFFSET \
// RUN:   -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DDIM=2 \
// RUN:   -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCube -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCube -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-NOTEXEL,CHECK-NOOFFSET,SPIRV-NOOFFSET \
// RUN:   -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=3 -DDIM=3
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCubeArray -DCOORD_DIM=4 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-NOTEXEL,CHECK-NOOFFSET,SPIRV-NOOFFSET \
// RUN:   -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=3 -DDIM=3
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-TEXEL,CHECK-OFFSET,DXIL-OFFSET \
// RUN:   -DDXIL_TY=7 -DRW=0 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-TEXEL,CHECK-OFFSET,SPIRV-OFFSET \
// RUN:   -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DDIM=2 \
// RUN:   -DOFFSET_CONST="<i32 1, i32 2>"

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   OFFSET_ARG         a literal offset argument
//   HAS_OFFSET         defined for types whose sampling and gathering methods
//                      have overloads taking an offset
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//   COORD_DIM          sample location components (DIM plus the array slice)
//   DXIL_TY            dx.Texture resource-kind operand
//   RW                 dx.Texture UAV operand
//   DIM                number of resource dimensions (offset, ddx/ddy, LOD
//                      location)
//   OFFSET_CONST       the offset literal as it appears in the IR
//   ARRAYED            spirv.Image Arrayed operand
//   SAMPLED            spirv.Image Sampled operand
//   IMG_FMT            spirv.Image Image Format operand
//   SPV_DIM            spirv.Image Dim operand
//
// Check prefixes:
//   TEXEL              the type has integer texel addressing (Load,
//                      operator[], mips), and therefore a `mips` field in its
//                      layout
//   OFFSET             the sampling and gathering methods have offset
//                      overloads
//   NOTEXEL            the type has no integer texel addressing
//   NOOFFSET           the sampling methods have no offset overloads, so
//                      their clamp overload takes the offset's place

// DXIL-TEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// DXIL-NOTEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) }
// DXIL: %"class.hlsl::SamplerState" = type { target("dx.Sampler", 0) }

// SPIRV-TEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// SPIRV-NOTEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) }
// SPIRV: %"class.hlsl::SamplerState" = type { target("spirv.Sampler") }

TEXTURE<float4> t;
SamplerState s;

// CHECK: @test_bias(float vector[[[COORD_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00)
// CHECK: ret <4 x float> %[[CALL]]

float4 test_bias(COORD_TYPE loc : LOC) : SV_Target {
  return t.SampleBias(s, loc, 0.0f);
}

// CHECK: define linkonce_odr {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float)(
// CHECK: %[[THIS_VAL1:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL1]], i32 0, i32 0
// CHECK: %[[HANDLE1:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP1]]
// CHECK: %[[SAMPLER_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK: %[[SAMPLER_H1:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP1]]
// CHECK: %[[BIAS_CAST1:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplebias.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE1]], target("dx.Sampler", 0) %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %{{.*}}, float %[[BIAS_CAST1]], <[[DIM]] x i32> zeroinitializer) [ "convergencectrl"(token %0) ]
// SPIRV: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplebias.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE1]], target("spirv.Sampler") %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %{{.*}}, float %[[BIAS_CAST1]], <[[DIM]] x i32> zeroinitializer) [ "convergencectrl"(token %0) ]

// CHECK-OFFSET: @test_offset(float vector[[[COORD_DIM]]])
// CHECK-OFFSET: %[[CALL_OFFSET:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float, int vector[[[DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <[[DIM]] x i32> noundef [[OFFSET_CONST]])
// CHECK-OFFSET: ret <4 x float> %[[CALL_OFFSET]]


#ifdef HAS_OFFSET
float4 test_offset(COORD_TYPE loc : LOC) : SV_Target {
  return t.SampleBias(s, loc, 0.0f, OFFSET_ARG);
}
#endif

// CHECK-OFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float, int vector[[[DIM]]])(
// CHECK-OFFSET: %[[THIS_VAL2:.*]] = load ptr, ptr %{{.*}}
// CHECK-OFFSET: %[[HANDLE_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL2]], i32 0, i32 0
// CHECK-OFFSET: %[[HANDLE2:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP2]]
// CHECK-OFFSET: %[[SAMPLER_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK-OFFSET: %[[SAMPLER_H2:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP2]]
// CHECK-OFFSET: %[[BIAS_CAST2:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL-OFFSET: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplebias.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]])
// SPIRV-OFFSET: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplebias.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]])

// CHECK-OFFSET: @test_clamp(float vector[[[COORD_DIM]]])
// CHECK-OFFSET: %[[CALL_CLAMP:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float, int vector[[[DIM]]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <[[DIM]] x i32> noundef [[OFFSET_CONST]], float {{.*}} 1.000000e+00)
// CHECK-OFFSET: ret <4 x float> %[[CALL_CLAMP]]

// CHECK-NOOFFSET: @test_clamp(
// CHECK-NOOFFSET: %[[CALL_NC:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float, float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, float {{.*}} 1.000000e+00)
// CHECK-NOOFFSET: ret <4 x float> %[[CALL_NC]]

#ifdef HAS_OFFSET
float4 test_clamp(COORD_TYPE loc : LOC) : SV_Target {
  return t.SampleBias(s, loc, 0.0f, OFFSET_ARG, 1.0f);
}
#else
// Cube textures have no offset overload, so the clamp takes the offset's place
// in the method signature; the intrinsic still receives a zero offset.
float4 test_clamp(COORD_TYPE loc : LOC) : SV_Target {
  return t.SampleBias(s, loc, 0.0f, 1.0f);
}
#endif

// CHECK-OFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float, int vector[[[DIM]]], float)(
// CHECK-OFFSET: %[[THIS_VAL3:.*]] = load ptr, ptr %{{.*}}
// CHECK-OFFSET: %[[HANDLE_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL3]], i32 0, i32 0
// CHECK-OFFSET: %[[HANDLE3:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP3]]
// CHECK-OFFSET: %[[SAMPLER_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK-OFFSET: %[[SAMPLER_H3:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP3]]
// CHECK-OFFSET: %[[BIAS_CAST3:.*]] = fptrunc {{.*}} double {{.*}} to float
// CHECK-OFFSET: %[[CLAMP_CAST3:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL-OFFSET: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplebias.clamp.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE3]], target("dx.Sampler", 0) %[[SAMPLER_H3]], <[[COORD_DIM]] x float> %{{.*}}, float %[[BIAS_CAST3]], <[[DIM]] x i32> %{{.*}}, float %[[CLAMP_CAST3]]) [ "convergencectrl"(token %0) ]
// SPIRV-OFFSET: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplebias.clamp.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE3]], target("spirv.Sampler") %[[SAMPLER_H3]], <[[COORD_DIM]] x float> %{{.*}}, float %[[BIAS_CAST3]], <[[DIM]] x i32> %{{.*}}, float %[[CLAMP_CAST3]]) [ "convergencectrl"(token %0) ]

// CHECK-NOOFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float, float)(
// CHECK-NOOFFSET: %[[THIS_VAL_NC:.*]] = load ptr, ptr %{{.*}}
// CHECK-NOOFFSET: %[[HANDLE_GEP_NC:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL_NC]], i32 0, i32 0
// CHECK-NOOFFSET: %[[HANDLE_NC:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP_NC]]
// CHECK-NOOFFSET: %[[SAMPLER_GEP_NC:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK-NOOFFSET: %[[SAMPLER_H_NC:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP_NC]]
// CHECK-NOOFFSET: %[[BIAS_CAST_NC:.*]] = fptrunc {{.*}} double {{.*}} to float
// CHECK-NOOFFSET: %[[CLAMP_CAST_NC:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL-NOOFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplebias.clamp.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE_NC]], target("dx.Sampler", 0) %[[SAMPLER_H_NC]], <[[COORD_DIM]] x float> %{{.*}}, float %[[BIAS_CAST_NC]], <[[DIM]] x i32> zeroinitializer, float %[[CLAMP_CAST_NC]])
// SPIRV-NOOFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplebias.clamp.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE_NC]], target("spirv.Sampler") %[[SAMPLER_H_NC]], <[[COORD_DIM]] x float> %{{.*}}, float %[[BIAS_CAST_NC]], <[[DIM]] x i32> zeroinitializer, float %[[CLAMP_CAST_NC]])
