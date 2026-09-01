// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 --check-prefixes=CHECK,DXIL -DDXIL_TY=2 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 --check-prefixes=CHECK,SPIRV -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 --check-prefixes=CHECK,DXIL -DDXIL_TY=7 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 --check-prefixes=CHECK,SPIRV -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0

TEXTURE<float4> t;
SamplerComparisonState s;

// CHECK: @test_cmp_level_zero(float vector[[[COORD_DIM]]], float)
// CHECK: %[[CALL:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmpLevelZero(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerComparisonState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00)
// CHECK: ret float %[[CALL]]

float test_cmp_level_zero(COORD_TYPE loc : LOC, float cmp : CMP) : SV_Target {
  return t.SampleCmpLevelZero(s, loc, 0.0f);
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmpLevelZero(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float)(
// CHECK-SAME: ptr noundef nonnull {{.*}} %[[THIS1:[^,]+]], ptr noundef byval(%"class.hlsl::SamplerComparisonState") {{.*}} %[[SAMPLER1:[^,]+]], <[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[COORD1:[^,]+]], float noundef nofpclass(nan inf) %[[CMP1:[^)]+]])
// CHECK: %[[THIS_VAL1:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL1]], i32 0, i32 0
// CHECK: %[[HANDLE1:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP1]]
// CHECK: %[[SAMPLER_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerComparisonState", ptr %[[SAMPLER1]], i32 0, i32 0
// CHECK: %[[SAMPLER_H1:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP1]]
// CHECK: %[[COORD_VAL1:.*]] = load <[[COORD_DIM]] x float>, ptr %{{.*}}
// CHECK: %[[CMP_VAL1:.*]] = load float, ptr %{{.*}}
// CHECK: %[[CMP_CAST1:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: call {{.*}} float @llvm.dx.resource.samplecmplevelzero.f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE1]], target("dx.Sampler", 0) %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %[[COORD_VAL1]], float %[[CMP_CAST1]], <2 x i32> zeroinitializer)
// SPIRV: call {{.*}} float @llvm.spv.resource.samplecmplevelzero.f32.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE1]], target("spirv.Sampler") %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %[[COORD_VAL1]], float %[[CMP_CAST1]], <2 x i32> zeroinitializer)

// CHECK: @test_cmp_level_zero_offset(float vector[[[COORD_DIM]]], float)
// CHECK: %[[CALL_OFFSET:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmpLevelZero(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float, int vector[2])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerComparisonState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <2 x i32> noundef <i32 1, i32 2>)
// CHECK: ret float %[[CALL_OFFSET]]

float test_cmp_level_zero_offset(COORD_TYPE loc : LOC, float cmp : CMP) : SV_Target {
  return t.SampleCmpLevelZero(s, loc, 0.0f, int2(1, 2));
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmpLevelZero(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float, int vector[2])(
// CHECK-SAME: ptr noundef nonnull {{.*}} %[[THIS2:[^,]+]], ptr noundef byval(%"class.hlsl::SamplerComparisonState") {{.*}} %[[SAMPLER2:[^,]+]], <[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[COORD2:[^,]+]], float noundef nofpclass(nan inf) %[[CMP2:[^,]+]], <2 x i32> noundef %[[OFFSET2:[^)]+]])
// CHECK: %[[THIS_VAL2:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL2]], i32 0, i32 0
// CHECK: %[[HANDLE2:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP2]]
// CHECK: %[[SAMPLER_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerComparisonState", ptr %[[SAMPLER2]], i32 0, i32 0
// CHECK: %[[SAMPLER_H2:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP2]]
// CHECK: %[[COORD_VAL2:.*]] = load <[[COORD_DIM]] x float>, ptr %{{.*}}
// CHECK: %[[CMP_VAL2:.*]] = load float, ptr %{{.*}}
// CHECK: %[[CMP_CAST2:.*]] = fptrunc {{.*}} double {{.*}} to float
// CHECK: %[[OFFSET_VAL2:.*]] = load <2 x i32>, ptr %{{.*}}
// DXIL: call {{.*}} float @llvm.dx.resource.samplecmplevelzero.f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE2]], target("dx.Sampler", 0) %[[SAMPLER_H2]], <[[COORD_DIM]] x float> %[[COORD_VAL2]], float %[[CMP_CAST2]], <2 x i32> %[[OFFSET_VAL2]])
// SPIRV: call {{.*}} float @llvm.spv.resource.samplecmplevelzero.f32.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE2]], target("spirv.Sampler") %[[SAMPLER_H2]], <[[COORD_DIM]] x float> %[[COORD_VAL2]], float %[[CMP_CAST2]], <2 x i32> %[[OFFSET_VAL2]])
