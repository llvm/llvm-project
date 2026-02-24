// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV

// DXIL: %"class.hlsl::Texture2D" = type { target("dx.Texture", <4 x float>, 0, 0, 0, 2) }
// DXIL: %"class.hlsl::SamplerState" = type { target("dx.Sampler", 0) }

// SPIRV: %"class.hlsl::Texture2D" = type { target("spirv.Image", float, 1, 2, 0, 0, 1, 0) }
// SPIRV: %"class.hlsl::SamplerState" = type { target("spirv.Sampler") }

Texture2D<float4> t;
SamplerState s;

// CHECK-LABEL: @test_level(float vector[2], float)
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::Texture2D<float vector[4]>::SampleLevel(hlsl::SamplerState, float vector[2], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <2 x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00)
// CHECK: ret <4 x float> %[[CALL]]

float4 test_level(float2 loc : LOC, float lod : LOD) : SV_Target {
  return t.SampleLevel(s, loc, 0.0f);
}

// CHECK-LABEL: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::Texture2D<float vector[4]>::SampleLevel(hlsl::SamplerState, float vector[2], float)(
// CHECK-SAME: ptr {{.*}} %[[THIS1:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER1:[^,]+]], <2 x float> noundef nofpclass(nan inf) %[[COORD1:[^,]+]], float noundef nofpclass(nan inf) %[[LOD1:[^)]+]])
// CHECK: %[[THIS_ADDR1:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR1:.*]] = alloca <2 x float>
// CHECK: %[[LOD_ADDR1:.*]] = alloca float
// CHECK: store ptr %[[THIS1]], ptr %[[THIS_ADDR1]]
// CHECK: store <2 x float> %[[COORD1]], ptr %[[COORD_ADDR1]]
// CHECK: store float %[[LOD1]], ptr %[[LOD_ADDR1]]
// CHECK: %[[THIS_VAL1:.*]] = load ptr, ptr %[[THIS_ADDR1]]
// CHECK: %[[HANDLE_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2D", ptr %[[THIS_VAL1]], i32 0, i32 0
// CHECK: %[[HANDLE1:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP1]]
// CHECK: %[[SAMPLER_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER1]], i32 0, i32 0
// CHECK: %[[SAMPLER_H1:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP1]]
// CHECK: %[[COORD_VAL1:.*]] = load <2 x float>, ptr %[[COORD_ADDR1]]
// CHECK: %[[LOD_VAL1:.*]] = load float, ptr %[[LOD_ADDR1]]
// CHECK: %[[LOD_CAST1:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: call {{.*}}                         <4 x float> @llvm.dx.resource.samplelevel.v4f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %[[HANDLE1]], target("dx.Sampler", 0) %[[SAMPLER_H1]], <2 x float> %[[COORD_VAL1]], float %[[LOD_CAST1]], <2 x i32> zeroinitializer)
// SPIRV: call {{.*}} <4 x float> @llvm.spv.resource.samplelevel.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE1]], target("spirv.Sampler") %[[SAMPLER_H1]], <2 x float> %[[COORD_VAL1]], float %[[LOD_CAST1]], <2 x i32> zeroinitializer)

// CHECK-LABEL: @test_offset(float vector[2], float)
// CHECK: %[[CALL_OFFSET:.*]] = call {{.*}} <4 x float> @hlsl::Texture2D<float vector[4]>::SampleLevel(hlsl::SamplerState, float vector[2], float, int vector[2])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <2 x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <2 x i32> noundef <i32 1, i32 2>)
// CHECK: ret <4 x float> %[[CALL_OFFSET]]

float4 test_offset(float2 loc : LOC, float lod : LOD) : SV_Target {
  return t.SampleLevel(s, loc, 0.0f, int2(1, 2));
}

// CHECK-LABEL: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::Texture2D<float vector[4]>::SampleLevel(hlsl::SamplerState, float vector[2], float, int vector[2])(
// CHECK-SAME: ptr {{.*}} %[[THIS2:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER2:[^,]+]], <2 x float> noundef nofpclass(nan inf) %[[COORD2:[^,]+]], float noundef nofpclass(nan inf) %[[LOD2:[^,]+]], <2 x i32> noundef %[[OFFSET2:[^)]+]])
// CHECK: %[[THIS_ADDR2:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR2:.*]] = alloca <2 x float>
// CHECK: %[[LOD_ADDR2:.*]] = alloca float
// CHECK: %[[OFFSET_ADDR2:.*]] = alloca <2 x i32>
// CHECK: store ptr %[[THIS2]], ptr %[[THIS_ADDR2]]
// CHECK: store <2 x float> %[[COORD2]], ptr %[[COORD_ADDR2]]
// CHECK: store float %[[LOD2]], ptr %[[LOD_ADDR2]]
// CHECK: store <2 x i32> %[[OFFSET2]], ptr %[[OFFSET_ADDR2]]
// CHECK: %[[THIS_VAL2:.*]] = load ptr, ptr %[[THIS_ADDR2]]
// CHECK: %[[HANDLE_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2D", ptr %[[THIS_VAL2]], i32 0, i32 0
// CHECK: %[[HANDLE2:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP2]]
// CHECK: %[[SAMPLER_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER2]], i32 0, i32 0
// CHECK: %[[SAMPLER_H2:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP2]]
// CHECK: %[[COORD_VAL2:.*]] = load <2 x float>, ptr %[[COORD_ADDR2]]
// CHECK: %[[LOD_VAL2:.*]] = load float, ptr %[[LOD_ADDR2]]
// CHECK: %[[LOD_CAST2:.*]] = fptrunc {{.*}} double {{.*}} to float
// CHECK: %[[OFFSET_VAL2:.*]] = load <2 x i32>, ptr %[[OFFSET_ADDR2]]
// DXIL: call {{.*}} <4 x float> @llvm.dx.resource.samplelevel.v4f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %[[HANDLE2]], target("dx.Sampler", 0) %[[SAMPLER_H2]], <2 x float> %[[COORD_VAL2]], float %[[LOD_CAST2]], <2 x i32> %[[OFFSET_VAL2]])
// SPIRV: call {{.*}} <4 x float> @llvm.spv.resource.samplelevel.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE2]], target("spirv.Sampler") %[[SAMPLER_H2]], <2 x float> %[[COORD_VAL2]], float %[[LOD_CAST2]], <2 x i32> %[[OFFSET_VAL2]])
