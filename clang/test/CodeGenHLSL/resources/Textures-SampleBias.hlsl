// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 --check-prefixes=CHECK,DXIL -DDXIL_TY=2 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 --check-prefixes=CHECK,SPIRV -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 --check-prefixes=CHECK,DXIL -DDXIL_TY=7 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 --check-prefixes=CHECK,SPIRV -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0

// DXIL: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// DXIL: %"class.hlsl::SamplerState" = type { target("dx.Sampler", 0) }

// SPIRV: %"class.hlsl::[[TEXTURE]]" = type { target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// SPIRV: %"class.hlsl::SamplerState" = type { target("spirv.Sampler") }

TEXTURE<float4> t;
SamplerState s;

// CHECK: @test_bias(float vector[[[COORD_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00)
// CHECK: ret <4 x float> %[[CALL]]

float4 test_bias(COORD_TYPE loc : LOC) : SV_Target {
  return t.SampleBias(s, loc, 0.0f);
}

// CHECK: define linkonce_odr {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float)(
// CHECK: %[[THIS_VAL1:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL1]], i32 0, i32 0
// CHECK: %[[HANDLE1:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP1]]
// CHECK: %[[SAMPLER_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK: %[[SAMPLER_H1:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP1]]
// CHECK: %[[BIAS_CAST1:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: %{{.*}} = call {{.*}} <4 x float> @llvm.dx.resource.samplebias.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE1]], target("dx.Sampler", 0) %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %{{.*}}, float %[[BIAS_CAST1]], <2 x i32> zeroinitializer)
// SPIRV: %{{.*}} = call {{.*}} <4 x float> @llvm.spv.resource.samplebias.v4f32.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE1]], target("spirv.Sampler") %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %{{.*}}, float %[[BIAS_CAST1]], <2 x i32> zeroinitializer)

// CHECK: @test_offset(float vector[[[COORD_DIM]]])
// CHECK: %[[CALL_OFFSET:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float, int vector[2])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <2 x i32> noundef <i32 1, i32 2>)
// CHECK: ret <4 x float> %[[CALL_OFFSET]]

float4 test_offset(COORD_TYPE loc : LOC) : SV_Target {
  return t.SampleBias(s, loc, 0.0f, int2(1, 2));
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float, int vector[2])(
// CHECK: %[[THIS_VAL2:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL2]], i32 0, i32 0
// CHECK: %[[HANDLE2:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP2]]
// CHECK: %[[SAMPLER_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK: %[[SAMPLER_H2:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP2]]
// CHECK: %[[BIAS_CAST2:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: %{{.*}} = call {{.*}} <4 x float> @llvm.dx.resource.samplebias.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]])
// SPIRV: %{{.*}} = call {{.*}} <4 x float> @llvm.spv.resource.samplebias.v4f32.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]])

// CHECK: @test_clamp(float vector[[[COORD_DIM]]])
// CHECK: %[[CALL_CLAMP:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float, int vector[2], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <2 x i32> noundef <i32 1, i32 2>, float {{.*}} 1.000000e+00)
// CHECK: ret <4 x float> %[[CALL_CLAMP]]

float4 test_clamp(COORD_TYPE loc : LOC) : SV_Target {
  return t.SampleBias(s, loc, 0.0f, int2(1, 2), 1.0f);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleBias(hlsl::SamplerState, float vector[[[COORD_DIM]]], float, int vector[2], float)(
// CHECK: %[[THIS_VAL3:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL3]], i32 0, i32 0
// CHECK: %[[HANDLE3:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP3]]
// CHECK: %[[SAMPLER_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK: %[[SAMPLER_H3:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP3]]
// CHECK: %[[BIAS_CAST3:.*]] = fptrunc {{.*}} double {{.*}} to float
// CHECK: %[[CLAMP_CAST3:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: %{{.*}} = call {{.*}} <4 x float> @llvm.dx.resource.samplebias.clamp.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE3]], target("dx.Sampler", 0) %[[SAMPLER_H3]], <[[COORD_DIM]] x float> %{{.*}}, float %[[BIAS_CAST3]], <2 x i32> %{{.*}}, float %[[CLAMP_CAST3]])
// SPIRV: %{{.*}} = call {{.*}} <4 x float> @llvm.spv.resource.samplebias.clamp.v4f32.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE3]], target("spirv.Sampler") %[[SAMPLER_H3]], <[[COORD_DIM]] x float> %{{.*}}, float %[[BIAS_CAST3]], <2 x i32> %{{.*}}, float %[[CLAMP_CAST3]])
