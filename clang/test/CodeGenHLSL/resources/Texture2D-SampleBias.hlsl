// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV

// DXIL: %"class.hlsl::Texture2D" = type { target("dx.Texture", <4 x float>, 0, 0, 0, 2) }
// DXIL: %"class.hlsl::SamplerState" = type { target("dx.Sampler", 0) }

// SPIRV: %"class.hlsl::Texture2D" = type { target("spirv.Image", float, 1, 2, 0, 0, 1, 0) }
// SPIRV: %"class.hlsl::SamplerState" = type { target("spirv.Sampler") }

Texture2D<float4> t;
SamplerState s;

// CHECK-LABEL: @test_bias(float vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::Texture2D<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[2], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <2 x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00)
// CHECK: ret <4 x float> %[[CALL]]

float4 test_bias(float2 loc : LOC) : SV_Target {
  return t.SampleBias(s, loc, 0.0f);
}

// CHECK-LABEL: define linkonce_odr {{.*}} <4 x float> @hlsl::Texture2D<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[2], float)(
// CHECK: %[[THIS_VAL1:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2D", ptr %[[THIS_VAL1]], i32 0, i32 0
// CHECK: %[[HANDLE1:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP1]]
// CHECK: %[[SAMPLER_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK: %[[SAMPLER_H1:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP1]]
// CHECK: %[[BIAS_CAST1:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: %{{.*}} = call {{.*}} <4 x float> @llvm.dx.resource.samplebias.v4f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %[[HANDLE1]], target("dx.Sampler", 0) %[[SAMPLER_H1]], <2 x float> %{{.*}}, float %[[BIAS_CAST1]], <2 x i32> zeroinitializer)
// SPIRV: %{{.*}} = call {{.*}} <4 x float> @llvm.spv.resource.samplebias.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE1]], target("spirv.Sampler") %[[SAMPLER_H1]], <2 x float> %{{.*}}, float %[[BIAS_CAST1]], <2 x i32> zeroinitializer)

// CHECK-LABEL: @test_offset(float vector[2])
// CHECK: %[[CALL_OFFSET:.*]] = call {{.*}} <4 x float> @hlsl::Texture2D<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[2], float, int vector[2])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <2 x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <2 x i32> noundef <i32 1, i32 2>)
// CHECK: ret <4 x float> %[[CALL_OFFSET]]

float4 test_offset(float2 loc : LOC) : SV_Target {
  return t.SampleBias(s, loc, 0.0f, int2(1, 2));
}

// CHECK-LABEL: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::Texture2D<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[2], float, int vector[2])(
// CHECK: %[[THIS_VAL2:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2D", ptr %[[THIS_VAL2]], i32 0, i32 0
// CHECK: %[[HANDLE2:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP2]]
// CHECK: %[[SAMPLER_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK: %[[SAMPLER_H2:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP2]]
// CHECK: %[[BIAS_CAST2:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: %{{.*}} = call {{.*}} <4 x float> @llvm.dx.resource.samplebias.v4f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %[[HANDLE2]], target("dx.Sampler", 0) %[[SAMPLER_H2]], <2 x float> %{{.*}}, float %[[BIAS_CAST2]], <2 x i32> %{{.*}})
// SPIRV: %{{.*}} = call {{.*}} <4 x float> @llvm.spv.resource.samplebias.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE2]], target("spirv.Sampler") %[[SAMPLER_H2]], <2 x float> %{{.*}}, float %[[BIAS_CAST2]], <2 x i32> %{{.*}})

// CHECK-LABEL: @test_clamp(float vector[2])
// CHECK: %[[CALL_CLAMP:.*]] = call {{.*}} <4 x float> @hlsl::Texture2D<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[2], float, int vector[2], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <2 x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <2 x i32> noundef <i32 1, i32 2>, float {{.*}} 1.000000e+00)
// CHECK: ret <4 x float> %[[CALL_CLAMP]]

float4 test_clamp(float2 loc : LOC) : SV_Target {
  return t.SampleBias(s, loc, 0.0f, int2(1, 2), 1.0f);
}

// CHECK-LABEL: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::Texture2D<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[2], float, int vector[2], float)(
// CHECK: %[[THIS_VAL3:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2D", ptr %[[THIS_VAL3]], i32 0, i32 0
// CHECK: %[[HANDLE3:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP3]]
// CHECK: %[[SAMPLER_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK: %[[SAMPLER_H3:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP3]]
// CHECK: %[[BIAS_CAST3:.*]] = fptrunc {{.*}} double {{.*}} to float
// CHECK: %[[CLAMP_CAST3:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: %{{.*}} = call {{.*}} <4 x float> @llvm.dx.resource.samplebias.clamp.v4f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %[[HANDLE3]], target("dx.Sampler", 0) %[[SAMPLER_H3]], <2 x float> %{{.*}}, float %[[BIAS_CAST3]], <2 x i32> %{{.*}}, float %[[CLAMP_CAST3]])
// SPIRV: %{{.*}} = call {{.*}} <4 x float> @llvm.spv.resource.samplebias.clamp.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE3]], target("spirv.Sampler") %[[SAMPLER_H3]], <2 x float> %{{.*}}, float %[[BIAS_CAST3]], <2 x i32> %{{.*}}, float %[[CLAMP_CAST3]])
