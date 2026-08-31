// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DGRAD_TYPE=float2 -DHAS_OFFSET \
// RUN:   -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-TEXEL,CHECK-OFFSET,DXIL-OFFSET \
// RUN:   -DDXIL_TY=2 -DRW=0 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DGRAD_TYPE=float3 -DTEXTURE=TextureCube -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCube -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-NOTEXEL,CHECK-NOOFFSET,DXIL-NOOFFSET -DDXIL_TY=5 -DRW=0 -DDIM=3
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DGRAD_TYPE=float2 -DHAS_OFFSET \
// RUN:   -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-TEXEL,CHECK-OFFSET,SPIRV-OFFSET \
// RUN:   -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DDIM=2 \
// RUN:   -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DGRAD_TYPE=float3 -DTEXTURE=TextureCube -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCube -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-NOTEXEL,CHECK-NOOFFSET,SPIRV-NOOFFSET -DARRAYED=0 -DSAMPLED=1 \
// RUN:   -DIMG_FMT=0 -DSPV_DIM=3 -DDIM=3
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DGRAD_TYPE=float2 -DHAS_OFFSET \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-TEXEL,CHECK-OFFSET,DXIL-OFFSET \
// RUN:   -DDXIL_TY=7 -DRW=0 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DGRAD_TYPE=float2 -DHAS_OFFSET \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-TEXEL,CHECK-OFFSET,SPIRV-OFFSET \
// RUN:   -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DDIM=2 \
// RUN:   -DOFFSET_CONST="<i32 1, i32 2>"

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   OFFSET_ARG         a literal offset argument
//   GRAD_TYPE          SampleGrad ddx/ddy type, one component per resource
//                      dimension
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

// CHECK: @test_grad(float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}})
// CHECK: ret <4 x float> %[[CALL]]

float4 test_grad(COORD_TYPE loc : LOC, GRAD_TYPE ddx : DDX, GRAD_TYPE ddy : DDY) : SV_Target {
  return t.SampleGrad(s, loc, ddx, ddy);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]])(
// CHECK-SAME: ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^,]+]], <[[DIM]] x float> {{.*}} %[[DDX:[^,]+]], <[[DIM]] x float> {{.*}} %[[DDY:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: %[[DDX_ADDR:.*]] = alloca <[[DIM]] x float>
// CHECK: %[[DDY_ADDR:.*]] = alloca <[[DIM]] x float>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: store <[[DIM]] x float> %[[DDX]], ptr %[[DDX_ADDR]]
// CHECK: store <[[DIM]] x float> %[[DDY]], ptr %[[DDY_ADDR]]
// CHECK: %[[THIS_VAL1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL1]], i32 0, i32 0
// CHECK: %[[HANDLE1:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP1]]
// CHECK: %[[SAMPLER_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H1:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP1]]
// CHECK: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// CHECK: %[[DDX_VAL:.*]] = load <[[DIM]] x float>, ptr %[[DDX_ADDR]]
// CHECK: %[[DDY_VAL:.*]] = load <[[DIM]] x float>, ptr %[[DDY_ADDR]]
// DXIL: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplegrad.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE1]], target("dx.Sampler", 0) %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x float> %[[DDX_VAL]], <[[DIM]] x float> %[[DDY_VAL]], <[[DIM]] x i32> zeroinitializer)
// SPIRV: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplegrad.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE1]], target("spirv.Sampler") %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x float> %[[DDX_VAL]], <[[DIM]] x float> %[[DDY_VAL]], <[[DIM]] x i32> zeroinitializer)

// CHECK-OFFSET: @test_offset(float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]])
// CHECK-OFFSET: %[[CALL_OFFSET:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]], int vector[[[DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x i32> noundef [[OFFSET_CONST]])
// CHECK-OFFSET: ret <4 x float> %[[CALL_OFFSET]]


#ifdef HAS_OFFSET
float4 test_offset(COORD_TYPE loc : LOC, GRAD_TYPE ddx : DDX, GRAD_TYPE ddy : DDY) : SV_Target {
  return t.SampleGrad(s, loc, ddx, ddy, OFFSET_ARG);
}
#endif

// CHECK-OFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]], int vector[[[DIM]]])(
// CHECK-OFFSET-SAME: ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^,]+]], <[[DIM]] x float> {{.*}} %[[DDX:[^,]+]], <[[DIM]] x float> {{.*}} %[[DDY:[^,]+]], <[[DIM]] x i32> {{.*}} %[[OFFSET:[^)]+]])
// CHECK-OFFSET: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK-OFFSET: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK-OFFSET: %[[DDX_ADDR:.*]] = alloca <[[DIM]] x float>
// CHECK-OFFSET: %[[DDY_ADDR:.*]] = alloca <[[DIM]] x float>
// CHECK-OFFSET: %[[OFFSET_ADDR:.*]] = alloca <[[DIM]] x i32>
// CHECK-OFFSET: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK-OFFSET: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK-OFFSET: store <[[DIM]] x float> %[[DDX]], ptr %[[DDX_ADDR]]
// CHECK-OFFSET: store <[[DIM]] x float> %[[DDY]], ptr %[[DDY_ADDR]]
// CHECK-OFFSET: store <[[DIM]] x i32> %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK-OFFSET: %[[THIS_VAL2:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK-OFFSET: %[[HANDLE_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL2]], i32 0, i32 0
// CHECK-OFFSET: %[[HANDLE2:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP2]]
// CHECK-OFFSET: %[[SAMPLER_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK-OFFSET: %[[SAMPLER_H2:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP2]]
// CHECK-OFFSET: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// CHECK-OFFSET: %[[DDX_VAL:.*]] = load <[[DIM]] x float>, ptr %[[DDX_ADDR]]
// CHECK-OFFSET: %[[DDY_VAL:.*]] = load <[[DIM]] x float>, ptr %[[DDY_ADDR]]
// CHECK-OFFSET: %[[OFFSET_VAL:.*]] = load <[[DIM]] x i32>, ptr %[[OFFSET_ADDR]]
// DXIL-OFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplegrad.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE2]], target("dx.Sampler", 0) %[[SAMPLER_H2]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x float> %[[DDX_VAL]], <[[DIM]] x float> %[[DDY_VAL]], <[[DIM]] x i32> %[[OFFSET_VAL]])
// SPIRV-OFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplegrad.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE2]], target("spirv.Sampler") %[[SAMPLER_H2]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x float> %[[DDX_VAL]], <[[DIM]] x float> %[[DDY_VAL]], <[[DIM]] x i32> %[[OFFSET_VAL]])

// CHECK-OFFSET: @test_clamp(float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]])
// CHECK-OFFSET: %[[CALL_CLAMP:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]], int vector[[[DIM]]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x i32> noundef [[OFFSET_CONST]], float {{.*}} 1.000000e+00)
// CHECK-OFFSET: ret <4 x float> %[[CALL_CLAMP]]

// CHECK-NOOFFSET: @test_clamp(
// CHECK-NOOFFSET: %[[CALL_NC:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 1.000000e+00)
// CHECK-NOOFFSET: ret <4 x float> %[[CALL_NC]]

#ifdef HAS_OFFSET
float4 test_clamp(COORD_TYPE loc : LOC, GRAD_TYPE ddx : DDX, GRAD_TYPE ddy : DDY) : SV_Target {
  return t.SampleGrad(s, loc, ddx, ddy, OFFSET_ARG, 1.0f);
}
#else
// Cube textures have no offset overload, so the clamp takes the offset's place
// in the method signature; the intrinsic still receives a zero offset.
float4 test_clamp(COORD_TYPE loc : LOC, GRAD_TYPE ddx : DDX, GRAD_TYPE ddy : DDY) : SV_Target {
  return t.SampleGrad(s, loc, ddx, ddy, 1.0f);
}
#endif

// CHECK-OFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]], int vector[[[DIM]]], float)(
// CHECK-OFFSET-SAME: ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^,]+]], <[[DIM]] x float> {{.*}} %[[DDX:[^,]+]], <[[DIM]] x float> {{.*}} %[[DDY:[^,]+]], <[[DIM]] x i32> {{.*}} %[[OFFSET:[^,]+]], float {{.*}} %[[CLAMP:[^)]+]])
// CHECK-OFFSET: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK-OFFSET: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK-OFFSET: %[[DDX_ADDR:.*]] = alloca <[[DIM]] x float>
// CHECK-OFFSET: %[[DDY_ADDR:.*]] = alloca <[[DIM]] x float>
// CHECK-OFFSET: %[[OFFSET_ADDR:.*]] = alloca <[[DIM]] x i32>
// CHECK-OFFSET: %[[CLAMP_ADDR:.*]] = alloca float
// CHECK-OFFSET: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK-OFFSET: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK-OFFSET: store <[[DIM]] x float> %[[DDX]], ptr %[[DDX_ADDR]]
// CHECK-OFFSET: store <[[DIM]] x float> %[[DDY]], ptr %[[DDY_ADDR]]
// CHECK-OFFSET: store <[[DIM]] x i32> %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK-OFFSET: store float %[[CLAMP]], ptr %[[CLAMP_ADDR]]
// CHECK-OFFSET: %[[THIS_VAL3:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK-OFFSET: %[[HANDLE_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL3]], i32 0, i32 0
// CHECK-OFFSET: %[[HANDLE3:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP3]]
// CHECK-OFFSET: %[[SAMPLER_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK-OFFSET: %[[SAMPLER_H3:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP3]]
// CHECK-OFFSET: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// CHECK-OFFSET: %[[DDX_VAL:.*]] = load <[[DIM]] x float>, ptr %[[DDX_ADDR]]
// CHECK-OFFSET: %[[DDY_VAL:.*]] = load <[[DIM]] x float>, ptr %[[DDY_ADDR]]
// CHECK-OFFSET: %[[OFFSET_VAL:.*]] = load <[[DIM]] x i32>, ptr %[[OFFSET_ADDR]]
// CHECK-OFFSET: %[[CLAMP_VAL:.*]] = load float, ptr %[[CLAMP_ADDR]]
// CHECK-OFFSET: %[[CLAMP_CAST3:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL-OFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplegrad.clamp.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE3]], target("dx.Sampler", 0) %[[SAMPLER_H3]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x float> %[[DDX_VAL]], <[[DIM]] x float> %[[DDY_VAL]], <[[DIM]] x i32> %[[OFFSET_VAL]], float %[[CLAMP_CAST3]])
// SPIRV-OFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplegrad.clamp.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE3]], target("spirv.Sampler") %[[SAMPLER_H3]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x float> %[[DDX_VAL]], <[[DIM]] x float> %[[DDY_VAL]], <[[DIM]] x i32> %[[OFFSET_VAL]], float %[[CLAMP_CAST3]])

// CHECK-NOOFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]], float)(
// CHECK-NOOFFSET: %[[THIS_VAL_NC:.*]] = load ptr, ptr %{{.*}}
// CHECK-NOOFFSET: %[[HANDLE_GEP_NC:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL_NC]], i32 0, i32 0
// CHECK-NOOFFSET: %[[HANDLE_NC:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP_NC]]
// CHECK-NOOFFSET: %[[SAMPLER_GEP_NC:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK-NOOFFSET: %[[SAMPLER_H_NC:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP_NC]]
// CHECK-NOOFFSET: %[[CLAMP_CAST_NC:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL-NOOFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplegrad.clamp.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE_NC]], target("dx.Sampler", 0) %[[SAMPLER_H_NC]], <[[COORD_DIM]] x float> %{{.*}}, <[[DIM]] x float> %{{.*}}, <[[DIM]] x float> %{{.*}}, <[[DIM]] x i32> zeroinitializer, float %[[CLAMP_CAST_NC]])
// SPIRV-NOOFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplegrad.clamp.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE_NC]], target("spirv.Sampler") %[[SAMPLER_H_NC]], <[[COORD_DIM]] x float> %{{.*}}, <[[DIM]] x float> %{{.*}}, <[[DIM]] x float> %{{.*}}, <[[DIM]] x i32> zeroinitializer, float %[[CLAMP_CAST_NC]])
