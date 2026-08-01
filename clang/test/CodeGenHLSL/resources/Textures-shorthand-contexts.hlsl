// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -emit-llvm -disable-llvm-passes -DTEXTURE=Texture2D -DCOORD_TYPE=float2 -o - %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 -DDXIL_TY=2 -DRW=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -emit-llvm -disable-llvm-passes -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 -o - %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 -DDXIL_TY=7 -DRW=0

// CHECK: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }

SamplerState g_s : register(s0);

struct S {
  TEXTURE tex;
};

// CHECK: define {{.*}}void @use_struct(S)(ptr noundef {{.*}}%s)
void use_struct(S s) {
  // CHECK: call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]])
  float4 val = s.tex.Sample(g_s, (COORD_TYPE)0.5);
}

// CHECK: define {{.*}}void @use_param(hlsl::[[TEXTURE]]<float vector[4]>)(ptr noundef {{.*}}%p)
void use_param(TEXTURE p) {
  // CHECK: call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]])
  float4 val = p.Sample(g_s, (COORD_TYPE)0.5);
}

[shader("pixel")]
float4 main() : SV_Target {
  // CHECK: %local = alloca %"class.hlsl::[[TEXTURE]]"
  TEXTURE local;
  // CHECK: call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]])
  return local.Sample(g_s, (COORD_TYPE)0.5);
}

// CHECK: declare <4 x float> @llvm.dx.resource.sample.v4f32
