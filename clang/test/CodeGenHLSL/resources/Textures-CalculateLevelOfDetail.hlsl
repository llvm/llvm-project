// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - -DTEXTURE=Texture2D %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2D --check-prefixes=CHECK,DXIL -DDXIL_TY=2 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - -DTEXTURE=Texture2D %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2D --check-prefixes=CHECK,SPIRV -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - -DTEXTURE=Texture2DArray %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2DArray --check-prefixes=CHECK,DXIL -DDXIL_TY=7 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - -DTEXTURE=Texture2DArray %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2DArray --check-prefixes=CHECK,SPIRV -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0

TEXTURE t;
SamplerState s;

// CHECK: define hidden {{.*}} float @test_lod(float vector[2])(<2 x float> {{.*}} %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::CalculateLevelOfDetail(hlsl::SamplerState, float vector[2])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <2 x float> {{.*}} %{{.*}})
// CHECK: ret float %[[CALL]]

float test_lod(float2 loc : LOC) : SV_Target {
  return t.CalculateLevelOfDetail(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::CalculateLevelOfDetail(hlsl::SamplerState, float vector[2])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <2 x float> {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <2 x float>, ptr %[[COORD]]
// DXIL: %[[RES:.*]] = call {{.*}} float @llvm.dx.resource.calculate.lod.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <2 x float> %[[COORD_VAL]])
// SPIRV: %[[RES:.*]] = call {{.*}} float @llvm.spv.resource.calculate.lod.f32.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <2 x float> %[[COORD_VAL]])
// CHECK: ret float %[[RES]]

// CHECK: define hidden {{.*}} float @test_lod_unclamped(float vector[2])(<2 x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[LOC_VAL:.*]] = load <2 x float>, ptr {{.*}}
// CHECK: %[[CALL:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::CalculateLevelOfDetailUnclamped(hlsl::SamplerState, float vector[2])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <2 x float> {{.*}} %[[LOC_VAL]])
// CHECK: ret float %[[CALL]]

float test_lod_unclamped(float2 loc : LOC) : SV_Target {
  return t.CalculateLevelOfDetailUnclamped(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::CalculateLevelOfDetailUnclamped(hlsl::SamplerState, float vector[2])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <2 x float> {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <2 x float>, ptr %[[COORD]]
// DXIL: %[[RES:.*]] = call {{.*}} float @llvm.dx.resource.calculate.lod.unclamped.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <2 x float> %[[COORD_VAL]])
// SPIRV: %[[RES:.*]] = call {{.*}} float @llvm.spv.resource.calculate.lod.unclamped.f32.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <2 x float> %[[COORD_VAL]])
// CHECK: ret float %[[RES]]
