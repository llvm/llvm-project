// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV

// DXIL: %"class.hlsl::Texture2D" = type { target("dx.Texture", <4 x float>, 0, 0, 0, 2) }
// DXIL: %"class.hlsl::SamplerComparisonState" = type { target("dx.Sampler", 0) }

// SPIRV: %"class.hlsl::Texture2D" = type { target("spirv.Image", float, 1, 2, 0, 0, 1, 0) }
// SPIRV: %"class.hlsl::SamplerComparisonState" = type { target("spirv.Sampler") }

Texture2D<float4> t;
SamplerComparisonState s;

// CHECK-LABEL: @test_cmp(float vector[2], float)
// CHECK: %[[CALL:.*]] = call {{.*}} float @hlsl::Texture2D<float vector[4]>::SampleCmp(hlsl::SamplerComparisonState, float vector[2], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerComparisonState") {{.*}}, <2 x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00)
// CHECK: ret float %[[CALL]]

float test_cmp(float2 loc : LOC, float cmp : CMP) : SV_Target {
  return t.SampleCmp(s, loc, 0.0f);
}

// CHECK-LABEL: define linkonce_odr hidden {{.*}} float @hlsl::Texture2D<float vector[4]>::SampleCmp(hlsl::SamplerComparisonState, float vector[2], float)(
// CHECK-SAME: ptr noundef nonnull {{.*}} %[[THIS1:[^,]+]], ptr noundef byval(%"class.hlsl::SamplerComparisonState") {{.*}} %[[SAMPLER1:[^,]+]], <2 x float> noundef nofpclass(nan inf) %[[COORD1:[^,]+]], float noundef nofpclass(nan inf) %[[CMP1:[^)]+]])
// CHECK: %[[THIS_VAL1:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2D", ptr %[[THIS_VAL1]], i32 0, i32 0
// CHECK: %[[HANDLE1:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP1]]
// CHECK: %[[SAMPLER_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerComparisonState", ptr %[[SAMPLER1]], i32 0, i32 0
// CHECK: %[[SAMPLER_H1:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP1]]
// CHECK: %[[COORD_VAL1:.*]] = load <2 x float>, ptr %{{.*}}
// CHECK: %[[CMP_VAL1:.*]] = load float, ptr %{{.*}}
// CHECK: %[[CMP_CAST1:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: call {{.*}} float @llvm.dx.resource.samplecmp.f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %[[HANDLE1]], target("dx.Sampler", 0) %[[SAMPLER_H1]], <2 x float> %[[COORD_VAL1]], float %[[CMP_CAST1]], <2 x i32> zeroinitializer)
// SPIRV: call {{.*}} float @llvm.spv.resource.samplecmp.f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE1]], target("spirv.Sampler") %[[SAMPLER_H1]], <2 x float> %[[COORD_VAL1]], float %[[CMP_CAST1]], <2 x i32> zeroinitializer)

// CHECK-LABEL: @test_offset(float vector[2], float)
// CHECK: %[[CALL_OFFSET:.*]] = call {{.*}} float @hlsl::Texture2D<float vector[4]>::SampleCmp(hlsl::SamplerComparisonState, float vector[2], float, int vector[2])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerComparisonState") {{.*}}, <2 x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <2 x i32> noundef <i32 1, i32 2>)
// CHECK: ret float %[[CALL_OFFSET]]

float test_offset(float2 loc : LOC, float cmp : CMP) : SV_Target {
  return t.SampleCmp(s, loc, 0.0f, int2(1, 2));
}

// CHECK-LABEL: define linkonce_odr hidden {{.*}} float @hlsl::Texture2D<float vector[4]>::SampleCmp(hlsl::SamplerComparisonState, float vector[2], float, int vector[2])(
// CHECK-SAME: ptr noundef nonnull {{.*}} %[[THIS2:[^,]+]], ptr noundef byval(%"class.hlsl::SamplerComparisonState") {{.*}} %[[SAMPLER2:[^,]+]], <2 x float> noundef nofpclass(nan inf) %[[COORD2:[^,]+]], float noundef nofpclass(nan inf) %[[CMP2:[^,]+]], <2 x i32> noundef %[[OFFSET2:[^)]+]])
// CHECK: %[[THIS_VAL2:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2D", ptr %[[THIS_VAL2]], i32 0, i32 0
// CHECK: %[[HANDLE2:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP2]]
// CHECK: %[[SAMPLER_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerComparisonState", ptr %[[SAMPLER2]], i32 0, i32 0
// CHECK: %[[SAMPLER_H2:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP2]]
// CHECK: %[[COORD_VAL2:.*]] = load <2 x float>, ptr %{{.*}}
// CHECK: %[[CMP_VAL2:.*]] = load float, ptr %{{.*}}
// CHECK: %[[CMP_CAST2:.*]] = fptrunc {{.*}} double {{.*}} to float
// CHECK: %[[OFFSET_VAL2:.*]] = load <2 x i32>, ptr %{{.*}}
// DXIL: call {{.*}} float @llvm.dx.resource.samplecmp.f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %[[HANDLE2]], target("dx.Sampler", 0) %[[SAMPLER_H2]], <2 x float> %[[COORD_VAL2]], float %[[CMP_CAST2]], <2 x i32> %[[OFFSET_VAL2]])
// SPIRV: call {{.*}} float @llvm.spv.resource.samplecmp.f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE2]], target("spirv.Sampler") %[[SAMPLER_H2]], <2 x float> %[[COORD_VAL2]], float %[[CMP_CAST2]], <2 x i32> %[[OFFSET_VAL2]])

// CHECK-LABEL: @test_clamp(float vector[2], float)
// CHECK: %[[CALL_CLAMP:.*]] = call {{.*}} float @hlsl::Texture2D<float vector[4]>::SampleCmp(hlsl::SamplerComparisonState, float vector[2], float, int vector[2], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerComparisonState") {{.*}}, <2 x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <2 x i32> noundef <i32 1, i32 2>, float {{.*}} 1.000000e+00)
// CHECK: ret float %[[CALL_CLAMP]]

float test_clamp(float2 loc : LOC, float cmp : CMP) : SV_Target {
  return t.SampleCmp(s, loc, 0.0f, int2(1, 2), 1.0f);
}

// CHECK-LABEL: define linkonce_odr hidden {{.*}} float @hlsl::Texture2D<float vector[4]>::SampleCmp(hlsl::SamplerComparisonState, float vector[2], float, int vector[2], float)(
// CHECK-SAME: ptr noundef nonnull {{.*}} %[[THIS3:[^,]+]], ptr noundef byval(%"class.hlsl::SamplerComparisonState") {{.*}} %[[SAMPLER3:[^,]+]], <2 x float> noundef nofpclass(nan inf) %[[COORD3:[^,]+]], float noundef nofpclass(nan inf) %[[CMP3:[^,]+]], <2 x i32> noundef %[[OFFSET3:[^,]+]], float noundef nofpclass(nan inf) %[[CLAMP3:[^)]+]])
// CHECK: %[[THIS_VAL3:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2D", ptr %[[THIS_VAL3]], i32 0, i32 0
// CHECK: %[[HANDLE3:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP3]]
// CHECK: %[[SAMPLER_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerComparisonState", ptr %[[SAMPLER3]], i32 0, i32 0
// CHECK: %[[SAMPLER_H3:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP3]]
// CHECK: %[[COORD_VAL3:.*]] = load <2 x float>, ptr %{{.*}}
// CHECK: %[[CMP_VAL3:.*]] = load float, ptr %{{.*}}
// CHECK: %[[CMP_CAST3:.*]] = fptrunc {{.*}} double {{.*}} to float
// CHECK: %[[OFFSET_VAL3:.*]] = load <2 x i32>, ptr %{{.*}}
// CHECK: %[[CLAMP_VAL3:.*]] = load float, ptr %{{.*}}
// CHECK: %[[CLAMP_CAST3:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: call {{.*}} float @llvm.dx.resource.samplecmp.clamp.f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %[[HANDLE3]], target("dx.Sampler", 0) %[[SAMPLER_H3]], <2 x float> %[[COORD_VAL3]], float %[[CMP_CAST3]], <2 x i32> %[[OFFSET_VAL3]], float %[[CLAMP_CAST3]])
// SPIRV: call {{.*}} float @llvm.spv.resource.samplecmp.clamp.f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE3]], target("spirv.Sampler") %[[SAMPLER_H3]], <2 x float> %[[COORD_VAL3]], float %[[CMP_CAST3]], <2 x i32> %[[OFFSET_VAL3]], float %[[CLAMP_CAST3]])
