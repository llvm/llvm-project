// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -emit-llvm -disable-llvm-passes \
// RUN:   -DTEXTURE=Texture2D -DCOORD_TYPE=float2 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 -DDXIL_TY=2 -DRW=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -emit-llvm -disable-llvm-passes \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 -DDXIL_TY=7 \
// RUN:   -DRW=0

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//   COORD_DIM          sample location components (DIM plus the array slice)
//   DXIL_TY            dx.Texture resource-kind operand
//   RW                 dx.Texture UAV operand

// CHECK: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }

SamplerState g_s : register(s0);

struct S {
  TEXTURE tex;
};

// CHECK: define {{.*}}void @use_struct(S)(ptr nofreeobj noundef {{.*}}%s)
void use_struct(S s) {
  // CHECK: call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, float vector[[[COORD_DIM]]])
  float4 val = s.tex.Sample(g_s, (COORD_TYPE)0.5);
}

// CHECK: define {{.*}}void @use_param(hlsl::[[TEXTURE]]<float vector[4]>)(ptr nofreeobj noundef {{.*}}%p)
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
