// Texture1D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float \
// RUN:   -DTEXTURE=Texture1D %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture1D --check-prefixes=CHECK,DXIL -DDXIL_TY=1 -DRW=0 \
// RUN:   -DGRAD_LLVM=float -DGRAD_CXX=float
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float \
// RUN:   -DTEXTURE=Texture1D %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture1D --check-prefixes=CHECK,SPIRV -DARRAYED=0 \
// RUN:   -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=0 -DGRAD_LLVM=float \
// RUN:   -DGRAD_CXX=float

// Texture1DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float \
// RUN:   -DTEXTURE=Texture1DArray %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture1DArray --check-prefixes=CHECK,DXIL -DDXIL_TY=6 \
// RUN:   -DRW=0 -DGRAD_LLVM=float -DGRAD_CXX=float
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float \
// RUN:   -DTEXTURE=Texture1DArray %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture1DArray --check-prefixes=CHECK,SPIRV -DARRAYED=1 \
// RUN:   -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=0 -DGRAD_LLVM=float \
// RUN:   -DGRAD_CXX=float

// Texture2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float2 \
// RUN:   -DTEXTURE=Texture2D %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture2D --check-prefixes=CHECK,DXIL -DDXIL_TY=2 -DRW=0 \
// RUN:   -DGRAD_LLVM="<2 x float>" -DGRAD_CXX="float vector[2]"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float2 \
// RUN:   -DTEXTURE=Texture2D %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture2D --check-prefixes=CHECK,SPIRV -DARRAYED=0 \
// RUN:   -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DGRAD_LLVM="<2 x float>" \
// RUN:   -DGRAD_CXX="float vector[2]"

// Texture2DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float2 \
// RUN:   -DTEXTURE=Texture2DArray %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture2DArray --check-prefixes=CHECK,DXIL -DDXIL_TY=7 \
// RUN:   -DRW=0 -DGRAD_LLVM="<2 x float>" -DGRAD_CXX="float vector[2]"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float2 \
// RUN:   -DTEXTURE=Texture2DArray %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture2DArray --check-prefixes=CHECK,SPIRV -DARRAYED=1 \
// RUN:   -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DGRAD_LLVM="<2 x float>" \
// RUN:   -DGRAD_CXX="float vector[2]"

// TextureCube
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float3 \
// RUN:   -DTEXTURE=TextureCube %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=TextureCube --check-prefixes=CHECK,DXIL -DDXIL_TY=5 -DRW=0 \
// RUN:   -DGRAD_LLVM="<3 x float>" -DGRAD_CXX="float vector[3]"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float3 \
// RUN:   -DTEXTURE=TextureCube %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=TextureCube --check-prefixes=CHECK,SPIRV -DARRAYED=0 \
// RUN:   -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=3 -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]"

// TextureCubeArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float3 \
// RUN:   -DTEXTURE=TextureCubeArray %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=TextureCubeArray --check-prefixes=CHECK,DXIL -DDXIL_TY=9 \
// RUN:   -DRW=0 -DGRAD_LLVM="<3 x float>" -DGRAD_CXX="float vector[3]"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float3 \
// RUN:   -DTEXTURE=TextureCubeArray %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=TextureCubeArray --check-prefixes=CHECK,SPIRV -DARRAYED=1 \
// RUN:   -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=3 -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]"

// Texture3D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float3 \
// RUN:   -DTEXTURE=Texture3D %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture3D --check-prefixes=CHECK,DXIL \
// RUN:   -DDXIL_TY=4 -DRW=0 -DDIM=3 \
// RUN:   -DGRAD_CXX="float vector[3]" -DGRAD_LLVM="<3 x float>"

// Texture3D
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DLOD_TYPE=float3 \
// RUN:   -DTEXTURE=Texture3D %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture3D --check-prefixes=CHECK,SPIRV \
// RUN:   -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=2 -DDIM=3 \
// RUN:   -DGRAD_CXX="float vector[3]" -DGRAD_LLVM="<3 x float>"

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   LOD_TYPE           CalculateLevelOfDetail location type
//   TEXTURE            resource type name
//   GRAD_CXX           ddx/ddy type in the C++ signature
//   GRAD_LLVM          ddx/ddy type in the IR
//   DXIL_TY            dx.Texture resource-kind operand
//   RW                 dx.Texture UAV operand
//   ARRAYED            spirv.Image Arrayed operand
//   SAMPLED            spirv.Image Sampled operand
//   IMG_FMT            spirv.Image Image Format operand
//   SPV_DIM            spirv.Image Dim operand

TEXTURE t;
SamplerState s;

// CHECK: define hidden {{.*}} float @test_lod([[GRAD_CXX]])([[GRAD_LLVM]] {{.*}} %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::CalculateLevelOfDetail(hlsl::SamplerState, [[GRAD_CXX]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, [[GRAD_LLVM]] {{.*}} %{{.*}})
// CHECK: ret float %[[CALL]]

float test_lod(LOD_TYPE loc : LOC) : SV_Target {
  return t.CalculateLevelOfDetail(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::CalculateLevelOfDetail(hlsl::SamplerState, [[GRAD_CXX]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], [[GRAD_LLVM]] {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load [[GRAD_LLVM]], ptr %[[COORD]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.calculate.lod.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], [[GRAD_LLVM]] %[[COORD_VAL]]) [ "convergencectrl"(token %0) ]
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.calculate.lod.f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], [[GRAD_LLVM]] %[[COORD_VAL]]) [ "convergencectrl"(token %0) ]
// CHECK: ret float %[[RES]]

// CHECK: define hidden {{.*}} float @test_lod_unclamped([[GRAD_CXX]])([[GRAD_LLVM]] noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[LOC_VAL:.*]] = load [[GRAD_LLVM]], ptr {{.*}}
// CHECK: %[[CALL:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::CalculateLevelOfDetailUnclamped(hlsl::SamplerState, [[GRAD_CXX]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, [[GRAD_LLVM]] {{.*}} %[[LOC_VAL]])
// CHECK: ret float %[[CALL]]

float test_lod_unclamped(LOD_TYPE loc : LOC) : SV_Target {
  return t.CalculateLevelOfDetailUnclamped(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::CalculateLevelOfDetailUnclamped(hlsl::SamplerState, [[GRAD_CXX]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], [[GRAD_LLVM]] {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load [[GRAD_LLVM]], ptr %[[COORD]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.calculate.lod.unclamped.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], [[GRAD_LLVM]] %[[COORD_VAL]]) [ "convergencectrl"(token %0) ]
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.calculate.lod.unclamped.f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], [[GRAD_LLVM]] %[[COORD_VAL]]) [ "convergencectrl"(token %0) ]
// CHECK: ret float %[[RES]]
