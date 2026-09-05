// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DOFFSET_ARG="1" \
// RUN:   -DGRAD_TYPE=float -DHAS_OFFSET -DTEXTURE=Texture1D \
// RUN:   -DCOORD_TYPE=float %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture1D -DCOORD_DIM=1 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-TEXEL,CHECK-OFFSET,DXIL-OFFSET \
// RUN:   -DDXIL_TY=1 -DRW=0 -DDIM=1 -DOFFSET_CONST="1" -DCOORD_LLVM=float \
// RUN:   -DCOORD_CXX=float -DINDEX_LLVM=i32 \
// RUN:   -DINDEX_CXX_U="unsigned int" -DINDEX_CXX_I=int \
// RUN:   -DGRAD_LLVM=float -DGRAD_CXX=float -DOFFSET_LLVM=i32 \
// RUN:   -DOFFSET_CXX=int -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DOFFSET_ARG="1" \
// RUN:   -DGRAD_TYPE=float -DHAS_OFFSET -DTEXTURE=Texture1DArray \
// RUN:   -DCOORD_TYPE=float2 %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture1DArray -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-TEXEL,CHECK-OFFSET,DXIL-OFFSET \
// RUN:   -DDXIL_TY=6 -DRW=0 -DDIM=1 -DOFFSET_CONST="1" \
// RUN:   -DCOORD_LLVM="<2 x float>" -DCOORD_CXX="float vector[2]" \
// RUN:   -DINDEX_LLVM="<2 x i32>" -DINDEX_CXX_U="unsigned int vector[2]" \
// RUN:   -DINDEX_CXX_I="int vector[2]" -DGRAD_LLVM=float \
// RUN:   -DGRAD_CXX=float -DOFFSET_LLVM=i32 -DOFFSET_CXX=int \
// RUN:   -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DGRAD_TYPE=float2 -DHAS_OFFSET \
// RUN:   -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s | llvm-cxxfilt | \
// RUN:   FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-TEXEL,CHECK-OFFSET,DXIL-OFFSET \
// RUN:   -DDXIL_TY=2 -DRW=0 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>" \
// RUN:   -DCOORD_LLVM="<2 x float>" -DCOORD_CXX="float vector[2]" \
// RUN:   -DINDEX_LLVM="<2 x i32>" -DINDEX_CXX_U="unsigned int vector[2]" \
// RUN:   -DINDEX_CXX_I="int vector[2]" -DGRAD_LLVM="<2 x float>" \
// RUN:   -DGRAD_CXX="float vector[2]" -DOFFSET_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[2]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DGRAD_TYPE=float3 -DTEXTURE=TextureCube -DCOORD_TYPE=float3 %s | \
// RUN:   llvm-cxxfilt | FileCheck %s -DTEXTURE=TextureCube -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-NOTEXEL,CHECK-NOOFFSET,DXIL-NOOFFSET \
// RUN:   -DDXIL_TY=5 -DRW=0 -DDIM=3 -DCOORD_LLVM="<3 x float>" \
// RUN:   -DCOORD_CXX="float vector[3]" -DINDEX_LLVM="<3 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[3]" \
// RUN:   -DINDEX_CXX_I="int vector[3]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DGRAD_TYPE=float3 -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 %s \
// RUN:   | llvm-cxxfilt | FileCheck %s -DTEXTURE=TextureCubeArray \
// RUN:   -DCOORD_DIM=4 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-NOTEXEL,CHECK-NOOFFSET,DXIL-NOOFFSET \
// RUN:   -DDXIL_TY=9 -DRW=0 -DDIM=3 -DCOORD_LLVM="<4 x float>" \
// RUN:   -DCOORD_CXX="float vector[4]" -DINDEX_LLVM="<4 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[4]" \
// RUN:   -DINDEX_CXX_I="int vector[4]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DOFFSET_ARG="1" \
// RUN:   -DGRAD_TYPE=float -DHAS_OFFSET -DTEXTURE=Texture1D \
// RUN:   -DCOORD_TYPE=float %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture1D -DCOORD_DIM=1 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-TEXEL,CHECK-OFFSET,SPIRV-OFFSET \
// RUN:   -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=0 -DDIM=1 \
// RUN:   -DOFFSET_CONST="1" -DCOORD_LLVM=float -DCOORD_CXX=float \
// RUN:   -DINDEX_LLVM=i32 -DINDEX_CXX_U="unsigned int" \
// RUN:   -DINDEX_CXX_I=int -DGRAD_LLVM=float -DGRAD_CXX=float \
// RUN:   -DOFFSET_LLVM=i32 -DOFFSET_CXX=int -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - -DOFFSET_ARG="1" \
// RUN:   -DGRAD_TYPE=float -DHAS_OFFSET -DTEXTURE=Texture1DArray \
// RUN:   -DCOORD_TYPE=float2 %s | llvm-cxxfilt | FileCheck %s \
// RUN:   -DTEXTURE=Texture1DArray -DCOORD_DIM=2 \
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
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DGRAD_TYPE=float2 -DHAS_OFFSET \
// RUN:   -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s | llvm-cxxfilt | \
// RUN:   FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
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
// RUN:   -DGRAD_TYPE=float3 -DTEXTURE=TextureCube -DCOORD_TYPE=float3 %s | \
// RUN:   llvm-cxxfilt | FileCheck %s -DTEXTURE=TextureCube -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-NOTEXEL,CHECK-NOOFFSET,SPIRV-NOOFFSET \
// RUN:   -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=3 -DDIM=3 \
// RUN:   -DCOORD_LLVM="<3 x float>" -DCOORD_CXX="float vector[3]" \
// RUN:   -DINDEX_LLVM="<3 x i32>" -DINDEX_CXX_U="unsigned int vector[3]" \
// RUN:   -DINDEX_CXX_I="int vector[3]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DGRAD_TYPE=float3 -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 %s \
// RUN:   | llvm-cxxfilt | FileCheck %s -DTEXTURE=TextureCubeArray \
// RUN:   -DCOORD_DIM=4 \
// RUN:   --check-prefixes=CHECK,SPIRV,SPIRV-NOTEXEL,CHECK-NOOFFSET,SPIRV-NOOFFSET \
// RUN:   -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=3 -DDIM=3 \
// RUN:   -DCOORD_LLVM="<4 x float>" -DCOORD_CXX="float vector[4]" \
// RUN:   -DINDEX_LLVM="<4 x i32>" -DINDEX_CXX_U="unsigned int vector[4]" \
// RUN:   -DINDEX_CXX_I="int vector[4]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DGRAD_TYPE=float2 -DHAS_OFFSET \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 %s | llvm-cxxfilt | \
// RUN:   FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL,DXIL-TEXEL,CHECK-OFFSET,DXIL-OFFSET \
// RUN:   -DDXIL_TY=7 -DRW=0 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>" \
// RUN:   -DCOORD_LLVM="<3 x float>" -DCOORD_CXX="float vector[3]" \
// RUN:   -DINDEX_LLVM="<3 x i32>" -DINDEX_CXX_U="unsigned int vector[3]" \
// RUN:   -DINDEX_CXX_I="int vector[3]" -DGRAD_LLVM="<2 x float>" \
// RUN:   -DGRAD_CXX="float vector[2]" -DOFFSET_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[2]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DGRAD_TYPE=float2 -DHAS_OFFSET \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 %s | llvm-cxxfilt | \
// RUN:   FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
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
//   GRAD_TYPE          SampleGrad ddx/ddy type, one component per resource
//                      dimension
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

// CHECK: @test_grad([[COORD_CXX]], [[GRAD_CXX]], [[GRAD_CXX]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, [[COORD_CXX]], [[GRAD_CXX]], [[GRAD_CXX]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, [[COORD_LLVM]] {{.*}} %{{.*}}, [[GRAD_LLVM]] {{.*}} %{{.*}}, [[GRAD_LLVM]] {{.*}} %{{.*}})
// CHECK: ret <4 x float> %[[CALL]]

float4 test_grad(COORD_TYPE loc : LOC, GRAD_TYPE ddx : DDX, GRAD_TYPE ddy : DDY) : SV_Target {
  return t.SampleGrad(s, loc, ddx, ddy);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, [[COORD_CXX]], [[GRAD_CXX]], [[GRAD_CXX]])(
// CHECK-SAME: ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], [[COORD_LLVM]] {{.*}} %[[COORD:[^,]+]], [[GRAD_LLVM]] {{.*}} %[[DDX:[^,]+]], [[GRAD_LLVM]] {{.*}} %[[DDY:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca [[COORD_LLVM]]
// CHECK: %[[DDX_ADDR:.*]] = alloca [[GRAD_LLVM]]
// CHECK: %[[DDY_ADDR:.*]] = alloca [[GRAD_LLVM]]
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store [[COORD_LLVM]] %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: store [[GRAD_LLVM]] %[[DDX]], ptr %[[DDX_ADDR]]
// CHECK: store [[GRAD_LLVM]] %[[DDY]], ptr %[[DDY_ADDR]]
// CHECK: %[[THIS_VAL1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL1]], i32 0, i32 0
// CHECK: %[[HANDLE1:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP1]]
// CHECK: %[[SAMPLER_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H1:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP1]]
// CHECK: %[[COORD_VAL:.*]] = load [[COORD_LLVM]], ptr %[[COORD_ADDR]]
// CHECK: %[[DDX_VAL:.*]] = load [[GRAD_LLVM]], ptr %[[DDX_ADDR]]
// CHECK: %[[DDY_VAL:.*]] = load [[GRAD_LLVM]], ptr %[[DDY_ADDR]]
// DXIL: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplegrad.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE1]], target("dx.Sampler", 0) %[[SAMPLER_H1]], [[COORD_LLVM]] %[[COORD_VAL]], [[GRAD_LLVM]] %[[DDX_VAL]], [[GRAD_LLVM]] %[[DDY_VAL]], [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// SPIRV: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplegrad.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE1]], target("spirv.Sampler") %[[SAMPLER_H1]], [[COORD_LLVM]] %[[COORD_VAL]], [[GRAD_LLVM]] %[[DDX_VAL]], [[GRAD_LLVM]] %[[DDY_VAL]], [[OFFSET_LLVM]] [[OFFSET_ZERO]])

// CHECK-OFFSET: @test_offset([[COORD_CXX]], [[GRAD_CXX]], [[GRAD_CXX]])
// CHECK-OFFSET: %[[CALL_OFFSET:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, [[COORD_CXX]], [[GRAD_CXX]], [[GRAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, [[COORD_LLVM]] {{.*}} %{{.*}}, [[GRAD_LLVM]] {{.*}} %{{.*}}, [[GRAD_LLVM]] {{.*}} %{{.*}}, [[OFFSET_LLVM]] noundef [[OFFSET_CONST]])
// CHECK-OFFSET: ret <4 x float> %[[CALL_OFFSET]]


#ifdef HAS_OFFSET
float4 test_offset(COORD_TYPE loc : LOC, GRAD_TYPE ddx : DDX, GRAD_TYPE ddy : DDY) : SV_Target {
  return t.SampleGrad(s, loc, ddx, ddy, OFFSET_ARG);
}
#endif

// CHECK-OFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, [[COORD_CXX]], [[GRAD_CXX]], [[GRAD_CXX]], [[OFFSET_CXX]])(
// CHECK-OFFSET-SAME: ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], [[COORD_LLVM]] {{.*}} %[[COORD:[^,]+]], [[GRAD_LLVM]] {{.*}} %[[DDX:[^,]+]], [[GRAD_LLVM]] {{.*}} %[[DDY:[^,]+]], [[OFFSET_LLVM]] {{.*}} %[[OFFSET:[^)]+]])
// CHECK-OFFSET: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK-OFFSET: %[[COORD_ADDR:.*]] = alloca [[COORD_LLVM]]
// CHECK-OFFSET: %[[DDX_ADDR:.*]] = alloca [[GRAD_LLVM]]
// CHECK-OFFSET: %[[DDY_ADDR:.*]] = alloca [[GRAD_LLVM]]
// CHECK-OFFSET: %[[OFFSET_ADDR:.*]] = alloca [[OFFSET_LLVM]]
// CHECK-OFFSET: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK-OFFSET: store [[COORD_LLVM]] %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK-OFFSET: store [[GRAD_LLVM]] %[[DDX]], ptr %[[DDX_ADDR]]
// CHECK-OFFSET: store [[GRAD_LLVM]] %[[DDY]], ptr %[[DDY_ADDR]]
// CHECK-OFFSET: store [[OFFSET_LLVM]] %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK-OFFSET: %[[THIS_VAL2:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK-OFFSET: %[[HANDLE_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL2]], i32 0, i32 0
// CHECK-OFFSET: %[[HANDLE2:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP2]]
// CHECK-OFFSET: %[[SAMPLER_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK-OFFSET: %[[SAMPLER_H2:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP2]]
// CHECK-OFFSET: %[[COORD_VAL:.*]] = load [[COORD_LLVM]], ptr %[[COORD_ADDR]]
// CHECK-OFFSET: %[[DDX_VAL:.*]] = load [[GRAD_LLVM]], ptr %[[DDX_ADDR]]
// CHECK-OFFSET: %[[DDY_VAL:.*]] = load [[GRAD_LLVM]], ptr %[[DDY_ADDR]]
// CHECK-OFFSET: %[[OFFSET_VAL:.*]] = load [[OFFSET_LLVM]], ptr %[[OFFSET_ADDR]]
// DXIL-OFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplegrad.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE2]], target("dx.Sampler", 0) %[[SAMPLER_H2]], [[COORD_LLVM]] %[[COORD_VAL]], [[GRAD_LLVM]] %[[DDX_VAL]], [[GRAD_LLVM]] %[[DDY_VAL]], [[OFFSET_LLVM]] %[[OFFSET_VAL]])
// SPIRV-OFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplegrad.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE2]], target("spirv.Sampler") %[[SAMPLER_H2]], [[COORD_LLVM]] %[[COORD_VAL]], [[GRAD_LLVM]] %[[DDX_VAL]], [[GRAD_LLVM]] %[[DDY_VAL]], [[OFFSET_LLVM]] %[[OFFSET_VAL]])

// CHECK-OFFSET: @test_clamp([[COORD_CXX]], [[GRAD_CXX]], [[GRAD_CXX]])
// CHECK-OFFSET: %[[CALL_CLAMP:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, [[COORD_CXX]], [[GRAD_CXX]], [[GRAD_CXX]], [[OFFSET_CXX]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, [[COORD_LLVM]] {{.*}} %{{.*}}, [[GRAD_LLVM]] {{.*}} %{{.*}}, [[GRAD_LLVM]] {{.*}} %{{.*}}, [[OFFSET_LLVM]] noundef [[OFFSET_CONST]], float {{.*}} 1.000000e+00)
// CHECK-OFFSET: ret <4 x float> %[[CALL_CLAMP]]

// CHECK-NOOFFSET: @test_clamp(
// CHECK-NOOFFSET: %[[CALL_NC:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, [[COORD_CXX]], [[GRAD_CXX]], [[GRAD_CXX]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, [[COORD_LLVM]] {{.*}} %{{.*}}, [[GRAD_LLVM]] {{.*}} %{{.*}}, [[GRAD_LLVM]] {{.*}} %{{.*}}, float {{.*}} 1.000000e+00)
// CHECK-NOOFFSET: ret <4 x float> %[[CALL_NC]]

#ifdef HAS_OFFSET
float4 test_clamp(COORD_TYPE loc : LOC, GRAD_TYPE ddx : DDX, GRAD_TYPE ddy : DDY) : SV_Target {
  return t.SampleGrad(s, loc, ddx, ddy, OFFSET_ARG, 1.0f);
}
#else
// Cube textures have no offset overload, so the clamp takes the offset's place
// in the method signature; the intrinsic still receives a zero offset.
float4 test_clamp(COORD_TYPE loc : LOC, GRAD_TYPE ddx : DDX, GRAD_TYPE ddy : DDY) : SV_Target {
  return t.SampleGrad(s, loc, ddx, ddy, 1.0f);
}
#endif

// CHECK-OFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, [[COORD_CXX]], [[GRAD_CXX]], [[GRAD_CXX]], [[OFFSET_CXX]], float)(
// CHECK-OFFSET-SAME: ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], [[COORD_LLVM]] {{.*}} %[[COORD:[^,]+]], [[GRAD_LLVM]] {{.*}} %[[DDX:[^,]+]], [[GRAD_LLVM]] {{.*}} %[[DDY:[^,]+]], [[OFFSET_LLVM]] {{.*}} %[[OFFSET:[^,]+]], float {{.*}} %[[CLAMP:[^)]+]])
// CHECK-OFFSET: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK-OFFSET: %[[COORD_ADDR:.*]] = alloca [[COORD_LLVM]]
// CHECK-OFFSET: %[[DDX_ADDR:.*]] = alloca [[GRAD_LLVM]]
// CHECK-OFFSET: %[[DDY_ADDR:.*]] = alloca [[GRAD_LLVM]]
// CHECK-OFFSET: %[[OFFSET_ADDR:.*]] = alloca [[OFFSET_LLVM]]
// CHECK-OFFSET: %[[CLAMP_ADDR:.*]] = alloca float
// CHECK-OFFSET: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK-OFFSET: store [[COORD_LLVM]] %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK-OFFSET: store [[GRAD_LLVM]] %[[DDX]], ptr %[[DDX_ADDR]]
// CHECK-OFFSET: store [[GRAD_LLVM]] %[[DDY]], ptr %[[DDY_ADDR]]
// CHECK-OFFSET: store [[OFFSET_LLVM]] %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK-OFFSET: store float %[[CLAMP]], ptr %[[CLAMP_ADDR]]
// CHECK-OFFSET: %[[THIS_VAL3:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK-OFFSET: %[[HANDLE_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL3]], i32 0, i32 0
// CHECK-OFFSET: %[[HANDLE3:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP3]]
// CHECK-OFFSET: %[[SAMPLER_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK-OFFSET: %[[SAMPLER_H3:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP3]]
// CHECK-OFFSET: %[[COORD_VAL:.*]] = load [[COORD_LLVM]], ptr %[[COORD_ADDR]]
// CHECK-OFFSET: %[[DDX_VAL:.*]] = load [[GRAD_LLVM]], ptr %[[DDX_ADDR]]
// CHECK-OFFSET: %[[DDY_VAL:.*]] = load [[GRAD_LLVM]], ptr %[[DDY_ADDR]]
// CHECK-OFFSET: %[[OFFSET_VAL:.*]] = load [[OFFSET_LLVM]], ptr %[[OFFSET_ADDR]]
// CHECK-OFFSET: %[[CLAMP_VAL:.*]] = load float, ptr %[[CLAMP_ADDR]]
// DXIL-OFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplegrad.clamp.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE3]], target("dx.Sampler", 0) %[[SAMPLER_H3]], [[COORD_LLVM]] %[[COORD_VAL]], [[GRAD_LLVM]] %[[DDX_VAL]], [[GRAD_LLVM]] %[[DDY_VAL]], [[OFFSET_LLVM]] %[[OFFSET_VAL]], float %[[CLAMP_VAL]])
// SPIRV-OFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplegrad.clamp.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE3]], target("spirv.Sampler") %[[SAMPLER_H3]], [[COORD_LLVM]] %[[COORD_VAL]], [[GRAD_LLVM]] %[[DDX_VAL]], [[GRAD_LLVM]] %[[DDY_VAL]], [[OFFSET_LLVM]] %[[OFFSET_VAL]], float %[[CLAMP_VAL]])

// CHECK-NOOFFSET: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, [[COORD_CXX]], [[GRAD_CXX]], [[GRAD_CXX]], float)(
// CHECK-NOOFFSET: %[[THIS_VAL_NC:.*]] = load ptr, ptr %{{.*}}
// CHECK-NOOFFSET: %[[HANDLE_GEP_NC:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL_NC]], i32 0, i32 0
// CHECK-NOOFFSET: %[[HANDLE_NC:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP_NC]]
// CHECK-NOOFFSET: %[[SAMPLER_GEP_NC:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %{{.*}}, i32 0, i32 0
// CHECK-NOOFFSET: %[[SAMPLER_H_NC:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP_NC]]
// CHECK-NOOFFSET: %[[COORD_VAL_NC:.*]] = load [[COORD_LLVM]], ptr %{{.*}}
// CHECK-NOOFFSET: %[[CLAMP_VAL_NC:.*]] = load float, ptr %{{.*}}
// DXIL-NOOFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplegrad.clamp.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE_NC]], target("dx.Sampler", 0) %[[SAMPLER_H_NC]], [[COORD_LLVM]] %{{.*}}, [[GRAD_LLVM]] %{{.*}}, [[GRAD_LLVM]] %{{.*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]], float %[[CLAMP_VAL_NC]])
// SPIRV-NOOFFSET: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplegrad.clamp.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE_NC]], target("spirv.Sampler") %[[SAMPLER_H_NC]], [[COORD_LLVM]] %{{.*}}, [[GRAD_LLVM]] %{{.*}}, [[GRAD_LLVM]] %{{.*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]], float %[[CLAMP_VAL_NC]])
