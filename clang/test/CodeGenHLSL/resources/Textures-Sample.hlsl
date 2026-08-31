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
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-NOTEXEL,CHECK-NOOFFSET,DXIL-NOOFFSET -DDXIL_TY=5 -DRW=0 -DDIM=3
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
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-NOTEXEL,CHECK-NOOFFSET,SPIRV-NOOFFSET -DARRAYED=0 -DSAMPLED=1 \
// RUN:   -DIMG_FMT=0 -DSPV_DIM=3 -DDIM=3
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

// CHECK: define hidden {{.*}} <4 x float> @main(float vector[[[COORD_DIM]]])(<[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}})
// CHECK: ret <4 x float> %[[CALL]]

float4 main(COORD_TYPE loc : LOC) : SV_Target {
  return t.Sample(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.sample.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x i32> zeroinitializer) [ "convergencectrl"(token %0) ]
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.sample.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x i32> zeroinitializer) [ "convergencectrl"(token %0) ]
// CHECK: ret <4 x float> %[[RES]]

// CHECK-OFFSET: define hidden {{.*}} <4 x float> @test_offset(float vector[[[COORD_DIM]]])(<[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK-OFFSET: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x i32> {{.*}} [[OFFSET_CONST]])
// CHECK-OFFSET: ret <4 x float> %[[CALL]]


#ifdef HAS_OFFSET
float4 test_offset(COORD_TYPE loc : LOC) : SV_Target {
  return t.Sample(s, loc, OFFSET_ARG);
}
#endif

// CHECK-OFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^,]+]], <[[DIM]] x i32> {{.*}} %[[OFFSET:[^)]+]])
// CHECK-OFFSET: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK-OFFSET: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK-OFFSET: %[[OFFSET_ADDR:.*]] = alloca <[[DIM]] x i32>
// CHECK-OFFSET: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK-OFFSET: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK-OFFSET: store <[[DIM]] x i32> %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK-OFFSET: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK-OFFSET: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK-OFFSET: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK-OFFSET: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK-OFFSET: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK-OFFSET: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// CHECK-OFFSET: %[[OFFSET_VAL:.*]] = load <[[DIM]] x i32>, ptr %[[OFFSET_ADDR]]
// DXIL-OFFSET: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.sample.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x i32> %[[OFFSET_VAL]]) [ "convergencectrl"(token %0) ]
// SPIRV-OFFSET: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.sample.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x i32> %[[OFFSET_VAL]]) [ "convergencectrl"(token %0) ]
// CHECK-OFFSET: ret <4 x float> %[[RES]]

// CHECK-OFFSET: define hidden {{.*}} <4 x float> @test_clamp(float vector[[[COORD_DIM]]])(<[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK-OFFSET: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]], int vector[[[DIM]]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}}, <[[DIM]] x i32> {{.*}} [[OFFSET_CONST]], float {{.*}} 1.000000e+00)
// CHECK-OFFSET: ret <4 x float> %[[CALL]]

// CHECK-NOOFFSET: @test_clamp(
// CHECK-NOOFFSET: %[[CALL_NC:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}}, float {{.*}} 1.000000e+00)
// CHECK-NOOFFSET: ret <4 x float> %[[CALL_NC]]

#ifdef HAS_OFFSET
float4 test_clamp(COORD_TYPE loc : LOC) : SV_Target {
  return t.Sample(s, loc, OFFSET_ARG, 1.0f);
}
#else
// Cube textures have no offset overload, so the clamp takes the offset's place
// in the method signature; the intrinsic still receives a zero offset.
float4 test_clamp(COORD_TYPE loc : LOC) : SV_Target {
  return t.Sample(s, loc, 1.0f);
}
#endif

// CHECK-OFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]], int vector[[[DIM]]], float)(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^,]+]], <[[DIM]] x i32> {{.*}} %[[OFFSET:[^,]+]], float {{.*}} %[[CLAMP:[^)]+]])
// CHECK-OFFSET: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK-OFFSET: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK-OFFSET: %[[OFFSET_ADDR:.*]] = alloca <[[DIM]] x i32>
// CHECK-OFFSET: %[[CLAMP_ADDR:.*]] = alloca float
// CHECK-OFFSET: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK-OFFSET: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK-OFFSET: store <[[DIM]] x i32> %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK-OFFSET: store float %[[CLAMP]], ptr %[[CLAMP_ADDR]]
// CHECK-OFFSET: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK-OFFSET: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK-OFFSET: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK-OFFSET: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK-OFFSET: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK-OFFSET: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// CHECK-OFFSET: %[[OFFSET_VAL:.*]] = load <[[DIM]] x i32>, ptr %[[OFFSET_ADDR]]
// CHECK-OFFSET: %[[CLAMP_VAL:.*]] = load float, ptr %[[CLAMP_ADDR]]
// CHECK-OFFSET: %[[CLAMP_CAST:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL-OFFSET: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.sample.clamp.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x i32> %[[OFFSET_VAL]], float %[[CLAMP_CAST]]) [ "convergencectrl"(token %0) ]
// SPIRV-OFFSET: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.sample.clamp.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x i32> %[[OFFSET_VAL]], float %[[CLAMP_CAST]]) [ "convergencectrl"(token %0) ]
// CHECK-OFFSET: ret <4 x float> %[[RES]]

// CHECK-NOOFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]], float)(
// CHECK-NOOFFSET: %[[THIS_VAL_NC:.*]] = load ptr, ptr %{{.*}}
// CHECK-NOOFFSET: %[[HANDLE_GEP_NC:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL_NC]], i32 0, i32 0
// CHECK-NOOFFSET: %[[HANDLE_NC:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP_NC]]
// CHECK-NOOFFSET: %[[SAMPLER_GEP_NC:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK-NOOFFSET: %[[SAMPLER_H_NC:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP_NC]]
// CHECK-NOOFFSET: %[[CLAMP_CAST_NC:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL-NOOFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.sample.clamp.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE_NC]], target("dx.Sampler", 0) %[[SAMPLER_H_NC]], <[[COORD_DIM]] x float> %{{.*}}, <[[DIM]] x i32> zeroinitializer, float %[[CLAMP_CAST_NC]])
// SPIRV-NOOFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.sample.clamp.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE_NC]], target("spirv.Sampler") %[[SAMPLER_H_NC]], <[[COORD_DIM]] x float> %{{.*}}, <[[DIM]] x i32> zeroinitializer, float %[[CLAMP_CAST_NC]])
