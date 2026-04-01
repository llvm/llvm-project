// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s

// CHECK: %"class.hlsl::Texture2D" = type { target("dx.Texture", <4 x float>, 0, 0, 0, 2), %"struct.hlsl::Texture2D<>::mips_type" }

SamplerState g_s : register(s0);

struct S {
  Texture2D tex;
};

// CHECK: define {{.*}}void @use_struct(S)(ptr noundef {{.*}}%s)
void use_struct(S s) {
  // CHECK: call {{.*}} <4 x float> @hlsl::Texture2D<float vector[4]>::Sample(hlsl::SamplerState, float vector[2])
  float4 val = s.tex.Sample(g_s, float2(0.5, 0.5));
}

// CHECK: define {{.*}}void @use_param(hlsl::Texture2D<float vector[4]>)(ptr noundef {{.*}}%p)
void use_param(Texture2D p) {
  // CHECK: call {{.*}} <4 x float> @hlsl::Texture2D<float vector[4]>::Sample(hlsl::SamplerState, float vector[2])
  float4 val = p.Sample(g_s, float2(0.5, 0.5));
}

[shader("pixel")]
float4 main() : SV_Target {
  // CHECK: %local = alloca %"class.hlsl::Texture2D"
  Texture2D local;
  // CHECK: call {{.*}} <4 x float> @hlsl::Texture2D<float vector[4]>::Sample(hlsl::SamplerState, float vector[2])
  return local.Sample(g_s, float2(0.5, 0.5));
}

// CHECK: declare <4 x float> @llvm.dx.resource.sample.v4f32
