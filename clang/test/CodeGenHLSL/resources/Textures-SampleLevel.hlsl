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
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-NOTEXEL -DDXIL_TY=5 -DRW=0 -DDIM=3
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCubeArray -DCOORD_DIM=4 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-NOTEXEL -DDXIL_TY=9 -DRW=0 -DDIM=3
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
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-NOTEXEL -DARRAYED=0 -DSAMPLED=1 \
// RUN:   -DIMG_FMT=0 -DSPV_DIM=3 -DDIM=3
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCubeArray -DCOORD_DIM=4 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-NOTEXEL -DARRAYED=1 -DSAMPLED=1 \
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

// DXIL-TEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// DXIL-NOTEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) }
// DXIL: %"class.hlsl::SamplerState" = type { target("dx.Sampler", 0) }

// SPIRV-TEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// SPIRV-NOTEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) }
// SPIRV: %"class.hlsl::SamplerState" = type { target("spirv.Sampler") }

TEXTURE<float4> t;
SamplerState s;

// CHECK: @test_level(float vector[[[COORD_DIM]]], float)
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleLevel(hlsl::SamplerState, float vector[[[COORD_DIM]]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00)
// CHECK: ret <4 x float> %[[CALL]]

float4 test_level(COORD_TYPE loc : LOC, float lod : LOD) : SV_Target {
  return t.SampleLevel(s, loc, 0.0f);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleLevel(hlsl::SamplerState, float vector[[[COORD_DIM]]], float)(
// CHECK-SAME: ptr {{.*}} %[[THIS1:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER1:[^,]+]], <[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[COORD1:[^,]+]], float noundef nofpclass(nan inf) %[[LOD1:[^)]+]])
// CHECK: %[[THIS_ADDR1:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR1:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: %[[LOD_ADDR1:.*]] = alloca float
// CHECK: store ptr %[[THIS1]], ptr %[[THIS_ADDR1]]
// CHECK: store <[[COORD_DIM]] x float> %[[COORD1]], ptr %[[COORD_ADDR1]]
// CHECK: store float %[[LOD1]], ptr %[[LOD_ADDR1]]
// CHECK: %[[THIS_VAL1:.*]] = load ptr, ptr %[[THIS_ADDR1]]
// CHECK: %[[HANDLE_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL1]], i32 0, i32 0
// CHECK: %[[HANDLE1:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP1]]
// CHECK: %[[SAMPLER_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER1]], i32 0, i32 0
// CHECK: %[[SAMPLER_H1:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP1]]
// CHECK: %[[COORD_VAL1:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR1]]
// CHECK: %[[LOD_VAL1:.*]] = load float, ptr %[[LOD_ADDR1]]
// CHECK: %[[LOD_CAST1:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: call reassoc nnan ninf nsz arcp afn                         <4 x float> @llvm.dx.resource.samplelevel.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE1]], target("dx.Sampler", 0) %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %[[COORD_VAL1]], float %[[LOD_CAST1]], <[[DIM]] x i32> zeroinitializer)
// SPIRV: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplelevel.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE1]], target("spirv.Sampler") %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %[[COORD_VAL1]], float %[[LOD_CAST1]], <[[DIM]] x i32> zeroinitializer)

// CHECK-OFFSET: @test_offset(float vector[[[COORD_DIM]]], float)
// CHECK-OFFSET: %[[CALL_OFFSET:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleLevel(hlsl::SamplerState, float vector[[[COORD_DIM]]], float, int vector[[[DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <[[DIM]] x i32> noundef [[OFFSET_CONST]])
// CHECK-OFFSET: ret <4 x float> %[[CALL_OFFSET]]

#ifdef HAS_OFFSET
float4 test_offset(COORD_TYPE loc : LOC, float lod : LOD) : SV_Target {
  return t.SampleLevel(s, loc, 0.0f, OFFSET_ARG);
}
#endif

// CHECK-OFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleLevel(hlsl::SamplerState, float vector[[[COORD_DIM]]], float, int vector[[[DIM]]])(
// CHECK-OFFSET-SAME: ptr {{.*}} %[[THIS2:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER2:[^,]+]], <[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[COORD2:[^,]+]], float noundef nofpclass(nan inf) %[[LOD2:[^,]+]], <[[DIM]] x i32> noundef %[[OFFSET2:[^)]+]])
// CHECK-OFFSET: %[[THIS_ADDR2:.*]] = alloca ptr
// CHECK-OFFSET: %[[COORD_ADDR2:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK-OFFSET: %[[LOD_ADDR2:.*]] = alloca float
// CHECK-OFFSET: %[[OFFSET_ADDR2:.*]] = alloca <[[DIM]] x i32>
// CHECK-OFFSET: store ptr %[[THIS2]], ptr %[[THIS_ADDR2]]
// CHECK-OFFSET: store <[[COORD_DIM]] x float> %[[COORD2]], ptr %[[COORD_ADDR2]]
// CHECK-OFFSET: store float %[[LOD2]], ptr %[[LOD_ADDR2]]
// CHECK-OFFSET: store <[[DIM]] x i32> %[[OFFSET2]], ptr %[[OFFSET_ADDR2]]
// CHECK-OFFSET: %[[THIS_VAL2:.*]] = load ptr, ptr %[[THIS_ADDR2]]
// CHECK-OFFSET: %[[HANDLE_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL2]], i32 0, i32 0
// CHECK-OFFSET: %[[HANDLE2:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP2]]
// CHECK-OFFSET: %[[SAMPLER_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER2]], i32 0, i32 0
// CHECK-OFFSET: %[[SAMPLER_H2:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP2]]
// CHECK-OFFSET: %[[COORD_VAL2:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR2]]
// CHECK-OFFSET: %[[LOD_VAL2:.*]] = load float, ptr %[[LOD_ADDR2]]
// CHECK-OFFSET: %[[LOD_CAST2:.*]] = fptrunc {{.*}} double {{.*}} to float
// CHECK-OFFSET: %[[OFFSET_VAL2:.*]] = load <[[DIM]] x i32>, ptr %[[OFFSET_ADDR2]]
// DXIL-OFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplelevel.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE2]], target("dx.Sampler", 0) %[[SAMPLER_H2]], <[[COORD_DIM]] x float> %[[COORD_VAL2]], float %[[LOD_CAST2]], <[[DIM]] x i32> %[[OFFSET_VAL2]])
// SPIRV-OFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplelevel.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE2]], target("spirv.Sampler") %[[SAMPLER_H2]], <[[COORD_DIM]] x float> %[[COORD_VAL2]], float %[[LOD_CAST2]], <[[DIM]] x i32> %[[OFFSET_VAL2]])
