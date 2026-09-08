// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float2 \
// RUN:   -DTEXTURE=Texture2D %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D --check-prefixes=CHECK,DXIL \
// RUN:   -DDXIL_TY=2 -DRW=0 -DDIM=2
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float3 \
// RUN:   -DTEXTURE=TextureCube %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCube --check-prefixes=CHECK,DXIL \
// RUN:   -DDXIL_TY=5 -DRW=0 -DDIM=3
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float3 \
// RUN:   -DTEXTURE=TextureCubeArray %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCubeArray \
// RUN:   --check-prefixes=CHECK,DXIL -DDXIL_TY=9 -DRW=0 -DDIM=3
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float2 \
// RUN:   -DTEXTURE=Texture2D %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D --check-prefixes=CHECK,SPIRV \
// RUN:   -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DDIM=2
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float3 \
// RUN:   -DTEXTURE=TextureCube %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCube --check-prefixes=CHECK,SPIRV \
// RUN:   -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=3 -DDIM=3
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float3 \
// RUN:   -DTEXTURE=TextureCubeArray %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCubeArray \
// RUN:   --check-prefixes=CHECK,SPIRV -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0 \
// RUN:   -DSPV_DIM=3 -DDIM=3
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float2 \
// RUN:   -DTEXTURE=Texture2DArray %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray --check-prefixes=CHECK,DXIL \
// RUN:   -DDXIL_TY=7 -DRW=0 -DDIM=2
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float2 \
// RUN:   -DTEXTURE=Texture2DArray %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray --check-prefixes=CHECK,SPIRV \
// RUN:   -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DDIM=2

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   LOD_TYPE           CalculateLevelOfDetail location type
//   TEXTURE            resource type name
//   DXIL_TY            dx.Texture resource-kind operand
//   RW                 dx.Texture UAV operand
//   DIM                number of resource dimensions (offset, ddx/ddy, LOD
//                      location)
//   ARRAYED            spirv.Image Arrayed operand
//   SAMPLED            spirv.Image Sampled operand
//   IMG_FMT            spirv.Image Image Format operand
//   SPV_DIM            spirv.Image Dim operand

TEXTURE t;
SamplerState s;

// CHECK: define hidden {{.*}} float @test_lod(float vector[[[DIM]]])(<[[DIM]] x float> {{.*}} %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::CalculateLevelOfDetail(hlsl::SamplerState, float vector[[[DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[DIM]] x float> {{.*}} %{{.*}})
// CHECK: ret float %[[CALL]]

float test_lod(LOD_TYPE loc : LOC) : SV_Target {
  return t.CalculateLevelOfDetail(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::CalculateLevelOfDetail(hlsl::SamplerState, float vector[[[DIM]]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[DIM]] x float> {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <[[DIM]] x float>, ptr %[[COORD]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.calculate.lod.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <[[DIM]] x float> %[[COORD_VAL]]) [ "convergencectrl"(token %0) ]
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.calculate.lod.f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <[[DIM]] x float> %[[COORD_VAL]]) [ "convergencectrl"(token %0) ]
// CHECK: ret float %[[RES]]

// CHECK: define hidden {{.*}} float @test_lod_unclamped(float vector[[[DIM]]])(<[[DIM]] x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[LOC_VAL:.*]] = load <[[DIM]] x float>, ptr {{.*}}
// CHECK: %[[CALL:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::CalculateLevelOfDetailUnclamped(hlsl::SamplerState, float vector[[[DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[DIM]] x float> {{.*}} %[[LOC_VAL]])
// CHECK: ret float %[[CALL]]

float test_lod_unclamped(LOD_TYPE loc : LOC) : SV_Target {
  return t.CalculateLevelOfDetailUnclamped(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::CalculateLevelOfDetailUnclamped(hlsl::SamplerState, float vector[[[DIM]]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[DIM]] x float> {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <[[DIM]] x float>, ptr %[[COORD]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.calculate.lod.unclamped.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <[[DIM]] x float> %[[COORD_VAL]]) [ "convergencectrl"(token %0) ]
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.calculate.lod.unclamped.f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <[[DIM]] x float> %[[COORD_VAL]]) [ "convergencectrl"(token %0) ]
// CHECK: ret float %[[RES]]
