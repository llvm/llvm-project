// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL -DDXIL_TY=2 -DRW=0 -DDIM=2 \
// RUN:   -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,SPIRV -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 \
// RUN:   -DSPV_DIM=1 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL -DDXIL_TY=7 -DRW=0 -DDIM=2 \
// RUN:   -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,SPIRV -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0 \
// RUN:   -DSPV_DIM=1 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>"

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   OFFSET_ARG         a literal offset argument
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//   COORD_DIM          sample location components (DIM plus the array slice)
//   DXIL_TY            dx.Texture resource-kind operand
//   RW                 dx.Texture UAV operand
//   DIM                number of resource dimensions (offset, ddx/ddy, LOD
//                      location)
//   OFFSET_CONST       the offset literal as it appears in the IR
//   ARRAYED            spirv.Image Arrayed operand
//   SAMPLED            spirv.Image Sampled operand
//   IMG_FMT            spirv.Image Image Format operand
//   SPV_DIM            spirv.Image Dim operand

// DXIL: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// DXIL: %"class.hlsl::SamplerComparisonState" = type { target("dx.Sampler", 0) }

// SPIRV: %"class.hlsl::[[TEXTURE]]" = type { target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// SPIRV: %"class.hlsl::SamplerComparisonState" = type { target("spirv.Sampler") }

TEXTURE<float4> t;
SamplerComparisonState s;

// CHECK: @test_cmp(float vector[[[COORD_DIM]]], float)
// CHECK: %[[CALL:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmp(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerComparisonState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00)
// CHECK: ret float %[[CALL]]

float test_cmp(COORD_TYPE loc : LOC, float cmp : CMP) : SV_Target {
  return t.SampleCmp(s, loc, 0.0f);
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmp(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float)(
// CHECK-SAME: ptr noundef nonnull {{.*}} %[[THIS1:[^,]+]], ptr noundef byval(%"class.hlsl::SamplerComparisonState") {{.*}} %[[SAMPLER1:[^,]+]], <[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[COORD1:[^,]+]], float noundef nofpclass(nan inf) %[[CMP1:[^)]+]])
// CHECK: %[[THIS_VAL1:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL1]], i32 0, i32 0
// CHECK: %[[HANDLE1:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP1]]
// CHECK: %[[SAMPLER_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerComparisonState", ptr %[[SAMPLER1]], i32 0, i32 0
// CHECK: %[[SAMPLER_H1:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP1]]
// CHECK: %[[COORD_VAL1:.*]] = load <[[COORD_DIM]] x float>, ptr %{{.*}}
// CHECK: %[[CMP_VAL1:.*]] = load float, ptr %{{.*}}
// CHECK: %[[CMP_CAST1:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.samplecmp.f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE1]], target("dx.Sampler", 0) %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %[[COORD_VAL1]], float %[[CMP_CAST1]], <[[DIM]] x i32> zeroinitializer)
// SPIRV: call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.samplecmp.f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE1]], target("spirv.Sampler") %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %[[COORD_VAL1]], float %[[CMP_CAST1]], <[[DIM]] x i32> zeroinitializer)

// CHECK: @test_offset(float vector[[[COORD_DIM]]], float)
// CHECK: %[[CALL_OFFSET:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmp(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float, int vector[[[DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerComparisonState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <[[DIM]] x i32> noundef [[OFFSET_CONST]])
// CHECK: ret float %[[CALL_OFFSET]]

float test_offset(COORD_TYPE loc : LOC, float cmp : CMP) : SV_Target {
  return t.SampleCmp(s, loc, 0.0f, OFFSET_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmp(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float, int vector[[[DIM]]])(
// CHECK-SAME: ptr noundef nonnull {{.*}} %[[THIS2:[^,]+]], ptr noundef byval(%"class.hlsl::SamplerComparisonState") {{.*}} %[[SAMPLER2:[^,]+]], <[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[COORD2:[^,]+]], float noundef nofpclass(nan inf) %[[CMP2:[^,]+]], <[[DIM]] x i32> noundef %[[OFFSET2:[^)]+]])
// CHECK: %[[THIS_VAL2:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL2]], i32 0, i32 0
// CHECK: %[[HANDLE2:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP2]]
// CHECK: %[[SAMPLER_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerComparisonState", ptr %[[SAMPLER2]], i32 0, i32 0
// CHECK: %[[SAMPLER_H2:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP2]]
// CHECK: %[[COORD_VAL2:.*]] = load <[[COORD_DIM]] x float>, ptr %{{.*}}
// CHECK: %[[CMP_VAL2:.*]] = load float, ptr %{{.*}}
// CHECK: %[[CMP_CAST2:.*]] = fptrunc {{.*}} double {{.*}} to float
// CHECK: %[[OFFSET_VAL2:.*]] = load <[[DIM]] x i32>, ptr %{{.*}}
// DXIL: call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.samplecmp.f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE2]], target("dx.Sampler", 0) %[[SAMPLER_H2]], <[[COORD_DIM]] x float> %[[COORD_VAL2]], float %[[CMP_CAST2]], <[[DIM]] x i32> %[[OFFSET_VAL2]])
// SPIRV: call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.samplecmp.f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE2]], target("spirv.Sampler") %[[SAMPLER_H2]], <[[COORD_DIM]] x float> %[[COORD_VAL2]], float %[[CMP_CAST2]], <[[DIM]] x i32> %[[OFFSET_VAL2]])

// CHECK: @test_clamp(float vector[[[COORD_DIM]]], float)
// CHECK: %[[CALL_CLAMP:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmp(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float, int vector[[[DIM]]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerComparisonState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <[[DIM]] x i32> noundef [[OFFSET_CONST]], float {{.*}} 1.000000e+00)
// CHECK: ret float %[[CALL_CLAMP]]

float test_clamp(COORD_TYPE loc : LOC, float cmp : CMP) : SV_Target {
  return t.SampleCmp(s, loc, 0.0f, OFFSET_ARG, 1.0f);
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmp(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float, int vector[[[DIM]]], float)(
// CHECK-SAME: ptr noundef nonnull {{.*}} %[[THIS3:[^,]+]], ptr noundef byval(%"class.hlsl::SamplerComparisonState") {{.*}} %[[SAMPLER3:[^,]+]], <[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[COORD3:[^,]+]], float noundef nofpclass(nan inf) %[[CMP3:[^,]+]], <[[DIM]] x i32> noundef %[[OFFSET3:[^,]+]], float noundef nofpclass(nan inf) %[[CLAMP3:[^)]+]])
// CHECK: %[[THIS_VAL3:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL3]], i32 0, i32 0
// CHECK: %[[HANDLE3:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP3]]
// CHECK: %[[SAMPLER_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerComparisonState", ptr %[[SAMPLER3]], i32 0, i32 0
// CHECK: %[[SAMPLER_H3:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP3]]
// CHECK: %[[COORD_VAL3:.*]] = load <[[COORD_DIM]] x float>, ptr %{{.*}}
// CHECK: %[[CMP_VAL3:.*]] = load float, ptr %{{.*}}
// CHECK: %[[CMP_CAST3:.*]] = fptrunc {{.*}} double {{.*}} to float
// CHECK: %[[OFFSET_VAL3:.*]] = load <[[DIM]] x i32>, ptr %{{.*}}
// CHECK: %[[CLAMP_VAL3:.*]] = load float, ptr %{{.*}}
// CHECK: %[[CLAMP_CAST3:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.samplecmp.clamp.f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE3]], target("dx.Sampler", 0) %[[SAMPLER_H3]], <[[COORD_DIM]] x float> %[[COORD_VAL3]], float %[[CMP_CAST3]], <[[DIM]] x i32> %[[OFFSET_VAL3]], float %[[CLAMP_CAST3]])
// SPIRV: call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.samplecmp.clamp.f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE3]], target("spirv.Sampler") %[[SAMPLER_H3]], <[[COORD_DIM]] x float> %[[COORD_VAL3]], float %[[CMP_CAST3]], <[[DIM]] x i32> %[[OFFSET_VAL3]], float %[[CLAMP_CAST3]])
