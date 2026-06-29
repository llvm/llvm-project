// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV

// DXIL: %"class.hlsl::Texture2DArray" = type { target("dx.Texture", <4 x float>, 0, 0, 0, 7), %"struct.hlsl::Texture2DArray<>::mips_type" }
// DXIL: %"class.hlsl::SamplerState" = type { target("dx.Sampler", 0) }

// SPIRV: %"class.hlsl::Texture2DArray" = type { target("spirv.Image", float, 1, 2, 1, 0, 1, 0), %"struct.hlsl::Texture2DArray<>::mips_type" }
// SPIRV: %"class.hlsl::SamplerState" = type { target("spirv.Sampler") }

Texture2DArray<float4> t;
SamplerState s;

// CHECK: define hidden {{.*}} <4 x float> @main(float vector[3])(<3 x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::Sample(hlsl::SamplerState, float vector[3])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <3 x float> {{.*}} %{{.*}})
// CHECK: ret <4 x float> %[[CALL]]

float4 main(float3 loc : LOC) : SV_Target {
  return t.Sample(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::Sample(hlsl::SamplerState, float vector[3])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <3 x float> {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <3 x float>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <3 x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2DArray", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <3 x float>, ptr %[[COORD_ADDR]]
// DXIL: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.dx.resource.sample.v4f32.tdx.Texture_v4f32_0_0_0_7t.tdx.Sampler_0t.v3f32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 7) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <3 x float> %[[COORD_VAL]], <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.spv.resource.sample.v4f32.tspirv.Image_f32_1_2_1_0_1_0t.tspirv.Samplert.v3f32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <3 x float> %[[COORD_VAL]], <2 x i32> zeroinitializer)
// CHECK: ret <4 x float> %[[RES]]

// CHECK: define hidden {{.*}} <4 x float> @test_offset(float vector[3])(<3 x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::Sample(hlsl::SamplerState, float vector[3], int vector[2])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <3 x float> {{.*}} %{{.*}}, <2 x i32> {{.*}} <i32 1, i32 2>)
// CHECK: ret <4 x float> %[[CALL]]

float4 test_offset(float3 loc : LOC) : SV_Target {
  return t.Sample(s, loc, int2(1, 2));
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::Sample(hlsl::SamplerState, float vector[3], int vector[2])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <3 x float> {{.*}} %[[COORD:[^,]+]], <2 x i32> {{.*}} %[[OFFSET:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <3 x float>
// CHECK: %[[OFFSET_ADDR:.*]] = alloca <2 x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <3 x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: store <2 x i32> %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2DArray", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <3 x float>, ptr %[[COORD_ADDR]]
// CHECK: %[[OFFSET_VAL:.*]] = load <2 x i32>, ptr %[[OFFSET_ADDR]]
// DXIL: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.dx.resource.sample.v4f32.tdx.Texture_v4f32_0_0_0_7t.tdx.Sampler_0t.v3f32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 7) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <3 x float> %[[COORD_VAL]], <2 x i32> %[[OFFSET_VAL]])
// SPIRV: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.spv.resource.sample.v4f32.tspirv.Image_f32_1_2_1_0_1_0t.tspirv.Samplert.v3f32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <3 x float> %[[COORD_VAL]], <2 x i32> %[[OFFSET_VAL]])
// CHECK: ret <4 x float> %[[RES]]

// CHECK: define hidden {{.*}} <4 x float> @test_clamp(float vector[3])(<3 x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::Sample(hlsl::SamplerState, float vector[3], int vector[2], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <3 x float> {{.*}}, <2 x i32> {{.*}} <i32 1, i32 2>, float {{.*}} 1.000000e+00)
// CHECK: ret <4 x float> %[[CALL]]

float4 test_clamp(float3 loc : LOC) : SV_Target {
  return t.Sample(s, loc, int2(1, 2), 1.0f);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::Sample(hlsl::SamplerState, float vector[3], int vector[2], float)(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <3 x float> {{.*}} %[[COORD:[^,]+]], <2 x i32> {{.*}} %[[OFFSET:[^,]+]], float {{.*}} %[[CLAMP:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <3 x float>
// CHECK: %[[OFFSET_ADDR:.*]] = alloca <2 x i32>
// CHECK: %[[CLAMP_ADDR:.*]] = alloca float
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <3 x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: store <2 x i32> %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK: store float %[[CLAMP]], ptr %[[CLAMP_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2DArray", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <3 x float>, ptr %[[COORD_ADDR]]
// CHECK: %[[OFFSET_VAL:.*]] = load <2 x i32>, ptr %[[OFFSET_ADDR]]
// CHECK: %[[CLAMP_VAL:.*]] = load float, ptr %[[CLAMP_ADDR]]
// CHECK: %[[CLAMP_CAST:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.dx.resource.sample.clamp.v4f32.tdx.Texture_v4f32_0_0_0_7t.tdx.Sampler_0t.v3f32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 7) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <3 x float> %[[COORD_VAL]], <2 x i32> %[[OFFSET_VAL]], float %[[CLAMP_CAST]])
// SPIRV: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.spv.resource.sample.clamp.v4f32.tspirv.Image_f32_1_2_1_0_1_0t.tspirv.Samplert.v3f32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <3 x float> %[[COORD_VAL]], <2 x i32> %[[OFFSET_VAL]], float %[[CLAMP_CAST]])
// CHECK: ret <4 x float> %[[RES]]
