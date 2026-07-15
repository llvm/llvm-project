// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2D -DCOORD_TYPE=float2 -o - %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 -DDXIL_TY=2 -DRW=0 --check-prefixes=CHECK
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2D -DCOORD_TYPE=float2 -o - %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 -DARRAYED=0 --check-prefixes=SPIRV
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 -o - %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 -DDXIL_TY=7 -DRW=0 --check-prefixes=CHECK
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 -o - %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 -DARRAYED=1 --check-prefixes=SPIRV

TEXTURE<> default_template : register(t1, space2);
TEXTURE implicit_template : register(t0, space1);

// CHECK: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// SPIRV: %"class.hlsl::[[TEXTURE]]" = type { target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, 1, 0), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }

// CHECK: @{{.*}}default_template = internal global %"class.hlsl::[[TEXTURE]]" poison, align {{[0-9]+}}
// CHECK: @{{.*}}implicit_template = internal global %"class.hlsl::[[TEXTURE]]" poison, align {{[0-9]+}}
// SPIRV: @{{.*}}default_template = internal global %"class.hlsl::[[TEXTURE]]" poison, align {{[0-9]+}}
// SPIRV: @{{.*}}implicit_template = internal global %"class.hlsl::[[TEXTURE]]" poison, align {{[0-9]+}}

// Each texture is initialized from its explicit register/space binding:
//   default_template -> register(t1, space2)  =>  registerNo 1, spaceNo 2
//   implicit_template -> register(t0, space1)  =>  registerNo 0, spaceNo 1
// CHECK: call {{.*}} @hlsl::[[TEXTURE]]<float vector[4]>::__createFromBinding{{.*}}(ptr {{.*}}@{{.*}}default_template, i32 noundef 1, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @{{.*}})
// CHECK: call {{.*}} @hlsl::[[TEXTURE]]<float vector[4]>::__createFromBinding{{.*}}(ptr {{.*}}@{{.*}}implicit_template, i32 noundef 0, i32 noundef 1, i32 noundef 1, i32 noundef 0, ptr noundef @{{.*}})
// SPIRV: call {{.*}} @hlsl::[[TEXTURE]]<float vector[4]>::__createFromBinding{{.*}}(ptr {{.*}}@{{.*}}default_template, i32 noundef 1, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @{{.*}})
// SPIRV: call {{.*}} @hlsl::[[TEXTURE]]<float vector[4]>::__createFromBinding{{.*}}(ptr {{.*}}@{{.*}}implicit_template, i32 noundef 0, i32 noundef 1, i32 noundef 1, i32 noundef 0, ptr noundef @{{.*}})

SamplerState sampl : register(s0);

float4 main(COORD_TYPE uv : TEXCOORD) : SV_Target {
  // CHECK: call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]])
  // SPIRV: call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]])
  float4 r = implicit_template.Sample(sampl, uv);
  return r;
}
