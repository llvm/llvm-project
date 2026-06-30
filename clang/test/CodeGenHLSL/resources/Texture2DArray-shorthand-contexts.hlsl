// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s

// CHECK: %"class.hlsl::Texture2DArray" = type { target("dx.Texture", <4 x float>, 0, 0, 0, 7), %"struct.hlsl::Texture2DArray<>::mips_type" }

SamplerState g_s : register(s0);

struct S {
  Texture2DArray tex;
};

// CHECK: define {{.*}}void @use_struct(S)(ptr noundef {{.*}}%s)
void use_struct(S s) {
  // CHECK: call {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::Sample(hlsl::SamplerState, float vector[3])
  float4 val = s.tex.Sample(g_s, float3(0.5, 0.5, 0.1));
}

// CHECK: define {{.*}}void @use_param(hlsl::Texture2DArray<float vector[4]>)(ptr noundef {{.*}}%p)
void use_param(Texture2DArray p) {
  // CHECK: call {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::Sample(hlsl::SamplerState, float vector[3])
  float4 val = p.Sample(g_s, float3(0.5, 0.5, 0.1));
}

[shader("pixel")]
float4 main() : SV_Target {
  // CHECK: %local = alloca %"class.hlsl::Texture2DArray"
  Texture2DArray local;
  // CHECK: call {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::Sample(hlsl::SamplerState, float vector[3])
  return local.Sample(g_s, float3(0.5, 0.5, 0.1));
}

// CHECK: declare <4 x float> @llvm.dx.resource.sample.v4f32
