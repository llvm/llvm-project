// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL,CHECK-OFFSET,DXIL-OFFSET -DDXIL_TY=2 \
// RUN:   -DRW=0 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCube -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCube -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL -DDXIL_TY=5 -DRW=0 -DDIM=3
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,SPIRV,CHECK-OFFSET,SPIRV-OFFSET -DARRAYED=0 \
// RUN:   -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DDIM=2 \
// RUN:   -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCube -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=TextureCube -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,SPIRV -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 \
// RUN:   -DSPV_DIM=3 -DDIM=3
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL,CHECK-OFFSET,DXIL-OFFSET -DDXIL_TY=7 \
// RUN:   -DRW=0 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,SPIRV,CHECK-OFFSET,SPIRV-OFFSET -DARRAYED=1 \
// RUN:   -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DDIM=2 \
// RUN:   -DOFFSET_CONST="<i32 1, i32 2>"

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   OFFSET_ARG         a literal offset argument
//   HAS_OFFSET         defined for types whose sampling and gathering methods
//                      have overloads taking an offset
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
//
// Check prefixes:
//   OFFSET             the sampling and gathering methods have offset
//                      overloads

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
// DXIL: call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.samplecmplevelzero.f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE1]], target("dx.Sampler", 0) %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %[[COORD_VAL1]], float %[[CMP_CAST1]], <[[DIM]] x i32> zeroinitializer)
// SPIRV: call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.samplecmplevelzero.f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE1]], target("spirv.Sampler") %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %[[COORD_VAL1]], float %[[CMP_CAST1]], <[[DIM]] x i32> zeroinitializer)

// CHECK-OFFSET: @test_cmp_level_zero_offset(float vector[[[COORD_DIM]]], float)
// CHECK-OFFSET: %[[CALL_OFFSET:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmpLevelZero(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float, int vector[[[DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerComparisonState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, <[[DIM]] x i32> noundef [[OFFSET_CONST]])
// CHECK-OFFSET: ret float %[[CALL_OFFSET]]

#ifdef HAS_OFFSET
float test_cmp_level_zero_offset(COORD_TYPE loc : LOC, float cmp : CMP) : SV_Target {
  return t.SampleCmpLevelZero(s, loc, 0.0f, OFFSET_ARG);
}
#endif

// CHECK-OFFSET: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmpLevelZero(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float, int vector[[[DIM]]])(
// CHECK-OFFSET-SAME: ptr noundef nonnull {{.*}} %[[THIS2:[^,]+]], ptr noundef byval(%"class.hlsl::SamplerComparisonState") {{.*}} %[[SAMPLER2:[^,]+]], <[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[COORD2:[^,]+]], float noundef nofpclass(nan inf) %[[CMP2:[^,]+]], <[[DIM]] x i32> noundef %[[OFFSET2:[^)]+]])
// CHECK-OFFSET: %[[THIS_VAL2:.*]] = load ptr, ptr %{{.*}}
// CHECK-OFFSET: %[[HANDLE_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL2]], i32 0, i32 0
// CHECK-OFFSET: %[[HANDLE2:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP2]]
// CHECK-OFFSET: %[[SAMPLER_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerComparisonState", ptr %[[SAMPLER2]], i32 0, i32 0
// CHECK-OFFSET: %[[SAMPLER_H2:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP2]]
// CHECK-OFFSET: %[[COORD_VAL2:.*]] = load <[[COORD_DIM]] x float>, ptr %{{.*}}
// CHECK-OFFSET: %[[CMP_VAL2:.*]] = load float, ptr %{{.*}}
// CHECK-OFFSET: %[[CMP_CAST2:.*]] = fptrunc {{.*}} double {{.*}} to float
// CHECK-OFFSET: %[[OFFSET_VAL2:.*]] = load <[[DIM]] x i32>, ptr %{{.*}}
// DXIL-OFFSET: call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.samplecmplevelzero.f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE2]], target("dx.Sampler", 0) %[[SAMPLER_H2]], <[[COORD_DIM]] x float> %[[COORD_VAL2]], float %[[CMP_CAST2]], <[[DIM]] x i32> %[[OFFSET_VAL2]])
// SPIRV-OFFSET: call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.samplecmplevelzero.f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE2]], target("spirv.Sampler") %[[SAMPLER_H2]], <[[COORD_DIM]] x float> %[[COORD_VAL2]], float %[[CMP_CAST2]], <[[DIM]] x i32> %[[OFFSET_VAL2]])
