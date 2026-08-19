// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DOFFSET_ARG="1" \
// RUN:   -DHAS_OFFSET -DTEXTURE=Texture1D -DCOORD_TYPE=float %s | \
// RUN:   llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture1D -DCOORD_DIM=1 \
// RUN:   --check-prefixes=CHECK,DXIL,CHECK-OFFSET,DXIL-OFFSET -DDXIL_TY=1 \
// RUN:   -DRW=0 -DDIM=1 -DOFFSET_CONST="1" -DCOORD_LLVM=float \
// RUN:   -DCOORD_CXX=float -DINDEX_LLVM=i32 \
// RUN:   -DINDEX_CXX_U="unsigned int" -DINDEX_CXX_I=int \
// RUN:   -DGRAD_LLVM=float -DGRAD_CXX=float -DOFFSET_LLVM=i32 \
// RUN:   -DOFFSET_CXX=int -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DOFFSET_ARG="1" \
// RUN:   -DHAS_OFFSET -DTEXTURE=Texture1DArray -DCOORD_TYPE=float2 %s | \
// RUN:   llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture1DArray -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL,CHECK-OFFSET,DXIL-OFFSET -DDXIL_TY=6 \
// RUN:   -DRW=0 -DDIM=1 -DOFFSET_CONST="1" -DCOORD_LLVM="<2 x float>" \
// RUN:   -DCOORD_CXX="float vector[2]" -DINDEX_LLVM="<2 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[2]" \
// RUN:   -DINDEX_CXX_I="int vector[2]" -DGRAD_LLVM=float \
// RUN:   -DGRAD_CXX=float -DOFFSET_LLVM=i32 -DOFFSET_CXX=int \
// RUN:   -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL,CHECK-OFFSET,DXIL-OFFSET -DDXIL_TY=2 \
// RUN:   -DRW=0 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>" \
// RUN:   -DCOORD_LLVM="<2 x float>" -DCOORD_CXX="float vector[2]" \
// RUN:   -DINDEX_LLVM="<2 x i32>" -DINDEX_CXX_U="unsigned int vector[2]" \
// RUN:   -DINDEX_CXX_I="int vector[2]" -DGRAD_LLVM="<2 x float>" \
// RUN:   -DGRAD_CXX="float vector[2]" -DOFFSET_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[2]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCube -DCOORD_TYPE=float3 %s | llvm-cxxfilt | \
// RUN:   FileCheck %s -DTEXTURE=TextureCube -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL -DDXIL_TY=5 -DRW=0 -DDIM=3 \
// RUN:   -DCOORD_LLVM="<3 x float>" -DCOORD_CXX="float vector[3]" \
// RUN:   -DINDEX_LLVM="<3 x i32>" -DINDEX_CXX_U="unsigned int vector[3]" \
// RUN:   -DINDEX_CXX_I="int vector[3]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 %s | llvm-cxxfilt | \
// RUN:   FileCheck %s -DTEXTURE=TextureCubeArray -DCOORD_DIM=4 \
// RUN:   --check-prefixes=CHECK,DXIL -DDXIL_TY=9 -DRW=0 -DDIM=3 \
// RUN:   -DCOORD_LLVM="<4 x float>" -DCOORD_CXX="float vector[4]" \
// RUN:   -DINDEX_LLVM="<4 x i32>" -DINDEX_CXX_U="unsigned int vector[4]" \
// RUN:   -DINDEX_CXX_I="int vector[4]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DOFFSET_ARG="1" \
// RUN:   -DHAS_OFFSET -DTEXTURE=Texture1D -DCOORD_TYPE=float %s | \
// RUN:   llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture1D -DCOORD_DIM=1 \
// RUN:   --check-prefixes=CHECK,SPIRV,CHECK-OFFSET,SPIRV-OFFSET -DARRAYED=0 \
// RUN:   -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=0 -DDIM=1 -DOFFSET_CONST="1" \
// RUN:   -DCOORD_LLVM=float -DCOORD_CXX=float -DINDEX_LLVM=i32 \
// RUN:   -DINDEX_CXX_U="unsigned int" -DINDEX_CXX_I=int \
// RUN:   -DGRAD_LLVM=float -DGRAD_CXX=float -DOFFSET_LLVM=i32 \
// RUN:   -DOFFSET_CXX=int -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DOFFSET_ARG="1" \
// RUN:   -DHAS_OFFSET -DTEXTURE=Texture1DArray -DCOORD_TYPE=float2 %s | \
// RUN:   llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture1DArray -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,SPIRV,CHECK-OFFSET,SPIRV-OFFSET -DARRAYED=1 \
// RUN:   -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=0 -DDIM=1 -DOFFSET_CONST="1" \
// RUN:   -DCOORD_LLVM="<2 x float>" -DCOORD_CXX="float vector[2]" \
// RUN:   -DINDEX_LLVM="<2 x i32>" -DINDEX_CXX_U="unsigned int vector[2]" \
// RUN:   -DINDEX_CXX_I="int vector[2]" -DGRAD_LLVM=float \
// RUN:   -DGRAD_CXX=float -DOFFSET_LLVM=i32 -DOFFSET_CXX=int \
// RUN:   -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,SPIRV,CHECK-OFFSET,SPIRV-OFFSET -DARRAYED=0 \
// RUN:   -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DDIM=2 \
// RUN:   -DOFFSET_CONST="<i32 1, i32 2>" -DCOORD_LLVM="<2 x float>" \
// RUN:   -DCOORD_CXX="float vector[2]" -DINDEX_LLVM="<2 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[2]" \
// RUN:   -DINDEX_CXX_I="int vector[2]" -DGRAD_LLVM="<2 x float>" \
// RUN:   -DGRAD_CXX="float vector[2]" -DOFFSET_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[2]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCube -DCOORD_TYPE=float3 %s | llvm-cxxfilt | \
// RUN:   FileCheck %s -DTEXTURE=TextureCube -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,SPIRV -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 \
// RUN:   -DSPV_DIM=3 -DDIM=3 -DCOORD_LLVM="<3 x float>" \
// RUN:   -DCOORD_CXX="float vector[3]" -DINDEX_LLVM="<3 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[3]" \
// RUN:   -DINDEX_CXX_I="int vector[3]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 %s | llvm-cxxfilt | \
// RUN:   FileCheck %s -DTEXTURE=TextureCubeArray -DCOORD_DIM=4 \
// RUN:   --check-prefixes=CHECK,SPIRV -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0 \
// RUN:   -DSPV_DIM=3 -DDIM=3 -DCOORD_LLVM="<4 x float>" \
// RUN:   -DCOORD_CXX="float vector[4]" -DINDEX_LLVM="<4 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[4]" \
// RUN:   -DINDEX_CXX_I="int vector[4]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL,CHECK-OFFSET,DXIL-OFFSET -DDXIL_TY=7 \
// RUN:   -DRW=0 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>" \
// RUN:   -DCOORD_LLVM="<3 x float>" -DCOORD_CXX="float vector[3]" \
// RUN:   -DINDEX_LLVM="<3 x i32>" -DINDEX_CXX_U="unsigned int vector[3]" \
// RUN:   -DINDEX_CXX_I="int vector[3]" -DGRAD_LLVM="<2 x float>" \
// RUN:   -DGRAD_CXX="float vector[2]" -DOFFSET_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[2]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,SPIRV,CHECK-OFFSET,SPIRV-OFFSET -DARRAYED=1 \
// RUN:   -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DDIM=2 \
// RUN:   -DOFFSET_CONST="<i32 1, i32 2>" -DCOORD_LLVM="<3 x float>" \
// RUN:   -DCOORD_CXX="float vector[3]" -DINDEX_LLVM="<3 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[3]" \
// RUN:   -DINDEX_CXX_I="int vector[3]" -DGRAD_LLVM="<2 x float>" \
// RUN:   -DGRAD_CXX="float vector[2]" -DOFFSET_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[2]" -DOFFSET_ZERO=zeroinitializer

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
//   GRAD_CXX           ddx/ddy type in the C++ signature
//   COORD_CXX          sample location type in the C++ signature
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

// CHECK: @test_cmp_level_zero([[COORD_CXX]], float)
// CHECK: %[[CALL:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmpLevelZero(hlsl::SamplerComparisonState, [[COORD_CXX]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerComparisonState") {{.*}}, [[COORD_LLVM]] {{.*}} %{{.*}}, float {{.*}} 0.000000e+00)
// CHECK: ret float %[[CALL]]

float test_cmp_level_zero(COORD_TYPE loc : LOC, float cmp : CMP) : SV_Target {
  return t.SampleCmpLevelZero(s, loc, 0.0f);
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmpLevelZero(hlsl::SamplerComparisonState, [[COORD_CXX]], float)(
// CHECK-SAME: ptr noundef nonnull {{.*}} %[[THIS1:[^,]+]], ptr noundef byval(%"class.hlsl::SamplerComparisonState") {{.*}} %[[SAMPLER1:[^,]+]], [[COORD_LLVM]] noundef nofpclass(nan inf) %[[COORD1:[^,]+]], float noundef nofpclass(nan inf) %[[CMP1:[^)]+]])
// CHECK: %[[THIS_VAL1:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[HANDLE_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL1]], i32 0, i32 0
// CHECK: %[[HANDLE1:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP1]]
// CHECK: %[[SAMPLER_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerComparisonState", ptr %[[SAMPLER1]], i32 0, i32 0
// CHECK: %[[SAMPLER_H1:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP1]]
// CHECK: %[[COORD_VAL1:.*]] = load [[COORD_LLVM]], ptr %{{.*}}
// CHECK: %[[CMP_VAL1:.*]] = load float, ptr %{{.*}}
// DXIL: call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.samplecmplevelzero.f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE1]], target("dx.Sampler", 0) %[[SAMPLER_H1]], [[COORD_LLVM]] %[[COORD_VAL1]], float %[[CMP_VAL1]], [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// SPIRV: call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.samplecmplevelzero.f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE1]], target("spirv.Sampler") %[[SAMPLER_H1]], [[COORD_LLVM]] %[[COORD_VAL1]], float %[[CMP_VAL1]], [[OFFSET_LLVM]] [[OFFSET_ZERO]])

// CHECK-OFFSET: @test_cmp_level_zero_offset([[COORD_CXX]], float)
// CHECK-OFFSET: %[[CALL_OFFSET:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmpLevelZero(hlsl::SamplerComparisonState, [[COORD_CXX]], float, [[OFFSET_CXX]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerComparisonState") {{.*}}, [[COORD_LLVM]] {{.*}} %{{.*}}, float {{.*}} 0.000000e+00, [[OFFSET_LLVM]] noundef [[OFFSET_CONST]])
// CHECK-OFFSET: ret float %[[CALL_OFFSET]]

#ifdef HAS_OFFSET
float test_cmp_level_zero_offset(COORD_TYPE loc : LOC, float cmp : CMP) : SV_Target {
  return t.SampleCmpLevelZero(s, loc, 0.0f, OFFSET_ARG);
}
#endif

// CHECK-OFFSET: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float vector[4]>::SampleCmpLevelZero(hlsl::SamplerComparisonState, [[COORD_CXX]], float, [[OFFSET_CXX]])(
// CHECK-OFFSET-SAME: ptr noundef nonnull {{.*}} %[[THIS2:[^,]+]], ptr noundef byval(%"class.hlsl::SamplerComparisonState") {{.*}} %[[SAMPLER2:[^,]+]], [[COORD_LLVM]] noundef nofpclass(nan inf) %[[COORD2:[^,]+]], float noundef nofpclass(nan inf) %[[CMP2:[^,]+]], [[OFFSET_LLVM]] noundef %[[OFFSET2:[^)]+]])
// CHECK-OFFSET: %[[THIS_VAL2:.*]] = load ptr, ptr %{{.*}}
// CHECK-OFFSET: %[[HANDLE_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL2]], i32 0, i32 0
// CHECK-OFFSET: %[[HANDLE2:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP2]]
// CHECK-OFFSET: %[[SAMPLER_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerComparisonState", ptr %[[SAMPLER2]], i32 0, i32 0
// CHECK-OFFSET: %[[SAMPLER_H2:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP2]]
// CHECK-OFFSET: %[[COORD_VAL2:.*]] = load [[COORD_LLVM]], ptr %{{.*}}
// CHECK-OFFSET: %[[CMP_VAL2:.*]] = load float, ptr %{{.*}}
// CHECK-OFFSET: %[[OFFSET_VAL2:.*]] = load [[OFFSET_LLVM]], ptr %{{.*}}
// DXIL-OFFSET: call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.samplecmplevelzero.f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE2]], target("dx.Sampler", 0) %[[SAMPLER_H2]], [[COORD_LLVM]] %[[COORD_VAL2]], float %[[CMP_VAL2]], [[OFFSET_LLVM]] %[[OFFSET_VAL2]])
// SPIRV-OFFSET: call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.samplecmplevelzero.f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE2]], target("spirv.Sampler") %[[SAMPLER_H2]], [[COORD_LLVM]] %[[COORD_VAL2]], float %[[CMP_VAL2]], [[OFFSET_LLVM]] %[[OFFSET_VAL2]])
