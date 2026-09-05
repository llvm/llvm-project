// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DOFFSET_ARG="1" \
// RUN:   -DHAS_OFFSET -DTEXTURE=Texture1D -DCOORD_TYPE=float %s | \
// RUN:   llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture1D -DCOORD_DIM=1 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-TEXEL,CHECK-OFFSET,DXIL-OFFSET \
// RUN:   -DDXIL_TY=1 -DRW=0 -DDIM=1 -DOFFSET_CONST="1" -DCOORD_LLVM=float \
// RUN:   -DCOORD_CXX=float -DINDEX_LLVM=i32 \
// RUN:   -DINDEX_CXX_U="unsigned int" -DINDEX_CXX_I=int \
// RUN:   -DGRAD_LLVM=float -DGRAD_CXX=float -DOFFSET_LLVM=i32 \
// RUN:   -DOFFSET_CXX=int -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DOFFSET_ARG="1" \
// RUN:   -DHAS_OFFSET -DTEXTURE=Texture1DArray -DCOORD_TYPE=float2 %s | \
// RUN:   llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture1DArray -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-TEXEL,CHECK-OFFSET,DXIL-OFFSET \
// RUN:   -DDXIL_TY=6 -DRW=0 -DDIM=1 -DOFFSET_CONST="1" \
// RUN:   -DCOORD_LLVM="<2 x float>" -DCOORD_CXX="float vector[2]" \
// RUN:   -DINDEX_LLVM="<2 x i32>" -DINDEX_CXX_U="unsigned int vector[2]" \
// RUN:   -DINDEX_CXX_I="int vector[2]" -DGRAD_LLVM=float \
// RUN:   -DGRAD_CXX=float -DOFFSET_LLVM=i32 -DOFFSET_CXX=int \
// RUN:   -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-TEXEL,CHECK-OFFSET,DXIL-OFFSET \
// RUN:   -DDXIL_TY=2 -DRW=0 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>" \
// RUN:   -DCOORD_LLVM="<2 x float>" -DCOORD_CXX="float vector[2]" \
// RUN:   -DINDEX_LLVM="<2 x i32>" -DINDEX_CXX_U="unsigned int vector[2]" \
// RUN:   -DINDEX_CXX_I="int vector[2]" -DGRAD_LLVM="<2 x float>" \
// RUN:   -DGRAD_CXX="float vector[2]" -DOFFSET_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[2]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCube -DCOORD_TYPE=float3 %s | llvm-cxxfilt | \
// RUN:   FileCheck %s -DTEXTURE=TextureCube -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-NOTEXEL,CHECK-NOOFFSET,DXIL-NOOFFSET \
// RUN:   -DDXIL_TY=5 -DRW=0 -DDIM=3 -DCOORD_LLVM="<3 x float>" \
// RUN:   -DCOORD_CXX="float vector[3]" -DINDEX_LLVM="<3 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[3]" \
// RUN:   -DINDEX_CXX_I="int vector[3]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 %s | llvm-cxxfilt | \
// RUN:   FileCheck %s -DTEXTURE=TextureCubeArray -DCOORD_DIM=4 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-NOTEXEL,CHECK-NOOFFSET,DXIL-NOOFFSET \
// RUN:   -DDXIL_TY=9 -DRW=0 -DDIM=3 -DCOORD_LLVM="<4 x float>" \
// RUN:   -DCOORD_CXX="float vector[4]" -DINDEX_LLVM="<4 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[4]" \
// RUN:   -DINDEX_CXX_I="int vector[4]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DOFFSET_ARG="1" \
// RUN:   -DHAS_OFFSET -DTEXTURE=Texture1D -DCOORD_TYPE=float %s | \
// RUN:   llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture1D -DCOORD_DIM=1 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-TEXEL,CHECK-OFFSET,SPIRV-OFFSET \
// RUN:   -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=0 -DDIM=1 \
// RUN:   -DOFFSET_CONST="1" -DCOORD_LLVM=float -DCOORD_CXX=float \
// RUN:   -DINDEX_LLVM=i32 -DINDEX_CXX_U="unsigned int" \
// RUN:   -DINDEX_CXX_I=int -DGRAD_LLVM=float -DGRAD_CXX=float \
// RUN:   -DOFFSET_LLVM=i32 -DOFFSET_CXX=int -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DOFFSET_ARG="1" \
// RUN:   -DHAS_OFFSET -DTEXTURE=Texture1DArray -DCOORD_TYPE=float2 %s | \
// RUN:   llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture1DArray -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-TEXEL,CHECK-OFFSET,SPIRV-OFFSET \
// RUN:   -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=0 -DDIM=1 \
// RUN:   -DOFFSET_CONST="1" -DCOORD_LLVM="<2 x float>" \
// RUN:   -DCOORD_CXX="float vector[2]" -DINDEX_LLVM="<2 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[2]" \
// RUN:   -DINDEX_CXX_I="int vector[2]" -DGRAD_LLVM=float \
// RUN:   -DGRAD_CXX=float -DOFFSET_LLVM=i32 -DOFFSET_CXX=int \
// RUN:   -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-TEXEL,CHECK-OFFSET,SPIRV-OFFSET \
// RUN:   -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DDIM=2 \
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
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-NOTEXEL,CHECK-NOOFFSET,SPIRV-NOOFFSET \
// RUN:   -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=3 -DDIM=3 \
// RUN:   -DCOORD_LLVM="<3 x float>" -DCOORD_CXX="float vector[3]" \
// RUN:   -DINDEX_LLVM="<3 x i32>" -DINDEX_CXX_U="unsigned int vector[3]" \
// RUN:   -DINDEX_CXX_I="int vector[3]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 %s | llvm-cxxfilt | \
// RUN:   FileCheck %s -DTEXTURE=TextureCubeArray -DCOORD_DIM=4 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-NOTEXEL,CHECK-NOOFFSET,SPIRV-NOOFFSET \
// RUN:   -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=3 -DDIM=3 \
// RUN:   -DCOORD_LLVM="<4 x float>" -DCOORD_CXX="float vector[4]" \
// RUN:   -DINDEX_LLVM="<4 x i32>" -DINDEX_CXX_U="unsigned int vector[4]" \
// RUN:   -DINDEX_CXX_I="int vector[4]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DHAS_OFFSET -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-TEXEL,CHECK-OFFSET,DXIL-OFFSET \
// RUN:   -DDXIL_TY=7 -DRW=0 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>" \
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
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-TEXEL,CHECK-OFFSET,SPIRV-OFFSET \
// RUN:   -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1 -DDIM=2 \
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
//   TEXEL              the type has integer texel addressing (Load,
//                      operator[], mips), and therefore a `mips` field in its
//                      layout
//   OFFSET             the sampling and gathering methods have offset
//                      overloads
//   NOTEXEL            the type has no integer texel addressing
//   NOOFFSET           the sampling methods have no offset overloads, so
//                      their clamp overload takes the offset's place

// DXIL-TEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// DXIL-NOTEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) }
// DXIL: %"class.hlsl::SamplerState" = type { target("dx.Sampler", 0) }

// SPIRV-TEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// SPIRV-NOTEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) }
// SPIRV: %"class.hlsl::SamplerState" = type { target("spirv.Sampler") }

TEXTURE<float4> t;
SamplerState s;

// CHECK: define hidden {{.*}} <4 x float> @main([[COORD_CXX]])([[COORD_LLVM]] noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, [[COORD_CXX]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, [[COORD_LLVM]] {{.*}} %{{.*}})
// CHECK: ret <4 x float> %[[CALL]]

float4 main(COORD_TYPE loc : LOC) : SV_Target {
  return t.Sample(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, [[COORD_CXX]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], [[COORD_LLVM]] {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca [[COORD_LLVM]]
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store [[COORD_LLVM]] %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load [[COORD_LLVM]], ptr %[[COORD_ADDR]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.sample.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], [[COORD_LLVM]] %[[COORD_VAL]], [[OFFSET_LLVM]] [[OFFSET_ZERO]]) [ "convergencectrl"(token %0) ]
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.sample.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], [[COORD_LLVM]] %[[COORD_VAL]], [[OFFSET_LLVM]] [[OFFSET_ZERO]]) [ "convergencectrl"(token %0) ]
// CHECK: ret <4 x float> %[[RES]]

// CHECK-OFFSET: define hidden {{.*}} <4 x float> @test_offset([[COORD_CXX]])([[COORD_LLVM]] noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK-OFFSET: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, [[COORD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, [[COORD_LLVM]] {{.*}} %{{.*}}, [[OFFSET_LLVM]] {{.*}} [[OFFSET_CONST]])
// CHECK-OFFSET: ret <4 x float> %[[CALL]]


#ifdef HAS_OFFSET
float4 test_offset(COORD_TYPE loc : LOC) : SV_Target {
  return t.Sample(s, loc, OFFSET_ARG);
}
#endif

// CHECK-OFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, [[COORD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], [[COORD_LLVM]] {{.*}} %[[COORD:[^,]+]], [[OFFSET_LLVM]] {{.*}} %[[OFFSET:[^)]+]])
// CHECK-OFFSET: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK-OFFSET: %[[COORD_ADDR:.*]] = alloca [[COORD_LLVM]]
// CHECK-OFFSET: %[[OFFSET_ADDR:.*]] = alloca [[OFFSET_LLVM]]
// CHECK-OFFSET: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK-OFFSET: store [[COORD_LLVM]] %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK-OFFSET: store [[OFFSET_LLVM]] %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK-OFFSET: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK-OFFSET: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK-OFFSET: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK-OFFSET: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK-OFFSET: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK-OFFSET: %[[COORD_VAL:.*]] = load [[COORD_LLVM]], ptr %[[COORD_ADDR]]
// CHECK-OFFSET: %[[OFFSET_VAL:.*]] = load [[OFFSET_LLVM]], ptr %[[OFFSET_ADDR]]
// DXIL-OFFSET: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.sample.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], [[COORD_LLVM]] %[[COORD_VAL]], [[OFFSET_LLVM]] %[[OFFSET_VAL]]) [ "convergencectrl"(token %0) ]
// SPIRV-OFFSET: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.sample.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], [[COORD_LLVM]] %[[COORD_VAL]], [[OFFSET_LLVM]] %[[OFFSET_VAL]]) [ "convergencectrl"(token %0) ]
// CHECK-OFFSET: ret <4 x float> %[[RES]]

// CHECK-OFFSET: define hidden {{.*}} <4 x float> @test_clamp([[COORD_CXX]])([[COORD_LLVM]] noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK-OFFSET: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, [[COORD_CXX]], [[OFFSET_CXX]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, [[COORD_LLVM]] {{.*}}, [[OFFSET_LLVM]] {{.*}} [[OFFSET_CONST]], float {{.*}} 1.000000e+00)
// CHECK-OFFSET: ret <4 x float> %[[CALL]]

// CHECK-NOOFFSET: @test_clamp(
// CHECK-NOOFFSET: %[[CALL_NC:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, [[COORD_CXX]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, [[COORD_LLVM]] {{.*}}, float {{.*}} 1.000000e+00)
// CHECK-NOOFFSET: ret <4 x float> %[[CALL_NC]]

#ifdef HAS_OFFSET
float4 test_clamp(COORD_TYPE loc : LOC) : SV_Target {
  return t.Sample(s, loc, OFFSET_ARG, 1.0f);
}
#else
// Cube textures have no offset overload, so the clamp takes the offset's place
// in the method signature; the intrinsic still receives a zero offset.
float4 test_clamp(COORD_TYPE loc : LOC) : SV_Target {
  return t.Sample(s, loc, 1.0f);
}
#endif

// CHECK-OFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, [[COORD_CXX]], [[OFFSET_CXX]], float)(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], [[COORD_LLVM]] {{.*}} %[[COORD:[^,]+]], [[OFFSET_LLVM]] {{.*}} %[[OFFSET:[^,]+]], float {{.*}} %[[CLAMP:[^)]+]])
// CHECK-OFFSET: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK-OFFSET: %[[COORD_ADDR:.*]] = alloca [[COORD_LLVM]]
// CHECK-OFFSET: %[[OFFSET_ADDR:.*]] = alloca [[OFFSET_LLVM]]
// CHECK-OFFSET: %[[CLAMP_ADDR:.*]] = alloca float
// CHECK-OFFSET: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK-OFFSET: store [[COORD_LLVM]] %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK-OFFSET: store [[OFFSET_LLVM]] %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK-OFFSET: store float %[[CLAMP]], ptr %[[CLAMP_ADDR]]
// CHECK-OFFSET: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK-OFFSET: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK-OFFSET: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK-OFFSET: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK-OFFSET: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK-OFFSET: %[[COORD_VAL:.*]] = load [[COORD_LLVM]], ptr %[[COORD_ADDR]]
// CHECK-OFFSET: %[[OFFSET_VAL:.*]] = load [[OFFSET_LLVM]], ptr %[[OFFSET_ADDR]]
// CHECK-OFFSET: %[[CLAMP_VAL:.*]] = load float, ptr %[[CLAMP_ADDR]]
// DXIL-OFFSET: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.sample.clamp.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], [[COORD_LLVM]] %[[COORD_VAL]], [[OFFSET_LLVM]] %[[OFFSET_VAL]], float %[[CLAMP_VAL]]) [ "convergencectrl"(token %0) ]
// SPIRV-OFFSET: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.sample.clamp.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], [[COORD_LLVM]] %[[COORD_VAL]], [[OFFSET_LLVM]] %[[OFFSET_VAL]], float %[[CLAMP_VAL]]) [ "convergencectrl"(token %0) ]
// CHECK-OFFSET: ret <4 x float> %[[RES]]

// CHECK-NOOFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Sample(hlsl::SamplerState, [[COORD_CXX]], float)(
// CHECK-NOOFFSET: %[[THIS_VAL_NC:.*]] = load ptr, ptr %{{.*}}
// CHECK-NOOFFSET: %[[HANDLE_GEP_NC:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL_NC]], i32 0, i32 0
// CHECK-NOOFFSET: %[[HANDLE_NC:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP_NC]]
// CHECK-NOOFFSET: %[[SAMPLER_GEP_NC:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK-NOOFFSET: %[[SAMPLER_H_NC:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP_NC]]
// CHECK-NOOFFSET: %[[COORD_VAL_NC:.*]] = load [[COORD_LLVM]], ptr %{{.*}}
// CHECK-NOOFFSET: %[[CLAMP_VAL_NC:.*]] = load float, ptr %{{.*}}
// DXIL-NOOFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.sample.clamp.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE_NC]], target("dx.Sampler", 0) %[[SAMPLER_H_NC]], [[COORD_LLVM]] %{{.*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]], float %[[CLAMP_VAL_NC]])
// SPIRV-NOOFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.sample.clamp.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE_NC]], target("spirv.Sampler") %[[SAMPLER_H_NC]], [[COORD_LLVM]] %{{.*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]], float %[[CLAMP_VAL_NC]])
