// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DGRAD_TYPE=float2 -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL -DDXIL_TY=2 -DRW=0 -DDIM=2 \
// RUN:   -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DGRAD_TYPE=float2 -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,SPIRV -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 \
// RUN:   -DSPV_DIM=1 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DGRAD_TYPE=float2 \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL -DDXIL_TY=7 -DRW=0 -DDIM=2 \
// RUN:   -DOFFSET_CONST="<i32 1, i32 2>"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -o - \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DGRAD_TYPE=float2 \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,SPIRV -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0 \
// RUN:   -DSPV_DIM=1 -DDIM=2 -DOFFSET_CONST="<i32 1, i32 2>"

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   OFFSET_ARG         a literal offset argument
//   GRAD_TYPE          SampleGrad ddx/ddy type, one component per resource
//                      dimension
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
// DXIL: %"class.hlsl::SamplerState" = type { target("dx.Sampler", 0) }

// SPIRV: %"class.hlsl::[[TEXTURE]]" = type { target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// SPIRV: %"class.hlsl::SamplerState" = type { target("spirv.Sampler") }

TEXTURE<float4> t;
SamplerState s;

// CHECK: @test_grad(float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}})
// CHECK: ret <4 x float> %[[CALL]]

float4 test_grad(COORD_TYPE loc : LOC, GRAD_TYPE ddx : DDX, GRAD_TYPE ddy : DDY) : SV_Target {
  return t.SampleGrad(s, loc, ddx, ddy);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]])(
// CHECK-SAME: ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^,]+]], <[[DIM]] x float> {{.*}} %[[DDX:[^,]+]], <[[DIM]] x float> {{.*}} %[[DDY:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: %[[DDX_ADDR:.*]] = alloca <[[DIM]] x float>
// CHECK: %[[DDY_ADDR:.*]] = alloca <[[DIM]] x float>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: store <[[DIM]] x float> %[[DDX]], ptr %[[DDX_ADDR]]
// CHECK: store <[[DIM]] x float> %[[DDY]], ptr %[[DDY_ADDR]]
// CHECK: %[[THIS_VAL1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL1]], i32 0, i32 0
// CHECK: %[[HANDLE1:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP1]]
// CHECK: %[[SAMPLER_GEP1:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H1:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP1]]
// CHECK: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// CHECK: %[[DDX_VAL:.*]] = load <[[DIM]] x float>, ptr %[[DDX_ADDR]]
// CHECK: %[[DDY_VAL:.*]] = load <[[DIM]] x float>, ptr %[[DDY_ADDR]]
// DXIL: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplegrad.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE1]], target("dx.Sampler", 0) %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x float> %[[DDX_VAL]], <[[DIM]] x float> %[[DDY_VAL]], <[[DIM]] x i32> zeroinitializer)
// SPIRV: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplegrad.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE1]], target("spirv.Sampler") %[[SAMPLER_H1]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x float> %[[DDX_VAL]], <[[DIM]] x float> %[[DDY_VAL]], <[[DIM]] x i32> zeroinitializer)

// CHECK: @test_offset(float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]])
// CHECK: %[[CALL_OFFSET:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]], int vector[[[DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x i32> noundef [[OFFSET_CONST]])
// CHECK: ret <4 x float> %[[CALL_OFFSET]]

float4 test_offset(COORD_TYPE loc : LOC, GRAD_TYPE ddx : DDX, GRAD_TYPE ddy : DDY) : SV_Target {
  return t.SampleGrad(s, loc, ddx, ddy, OFFSET_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]], int vector[[[DIM]]])(
// CHECK-SAME: ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^,]+]], <[[DIM]] x float> {{.*}} %[[DDX:[^,]+]], <[[DIM]] x float> {{.*}} %[[DDY:[^,]+]], <[[DIM]] x i32> {{.*}} %[[OFFSET:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: %[[DDX_ADDR:.*]] = alloca <[[DIM]] x float>
// CHECK: %[[DDY_ADDR:.*]] = alloca <[[DIM]] x float>
// CHECK: %[[OFFSET_ADDR:.*]] = alloca <[[DIM]] x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: store <[[DIM]] x float> %[[DDX]], ptr %[[DDX_ADDR]]
// CHECK: store <[[DIM]] x float> %[[DDY]], ptr %[[DDY_ADDR]]
// CHECK: store <[[DIM]] x i32> %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK: %[[THIS_VAL2:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL2]], i32 0, i32 0
// CHECK: %[[HANDLE2:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP2]]
// CHECK: %[[SAMPLER_GEP2:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H2:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP2]]
// CHECK: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// CHECK: %[[DDX_VAL:.*]] = load <[[DIM]] x float>, ptr %[[DDX_ADDR]]
// CHECK: %[[DDY_VAL:.*]] = load <[[DIM]] x float>, ptr %[[DDY_ADDR]]
// CHECK: %[[OFFSET_VAL:.*]] = load <[[DIM]] x i32>, ptr %[[OFFSET_ADDR]]
// DXIL: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplegrad.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE2]], target("dx.Sampler", 0) %[[SAMPLER_H2]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x float> %[[DDX_VAL]], <[[DIM]] x float> %[[DDY_VAL]], <[[DIM]] x i32> %[[OFFSET_VAL]])
// SPIRV: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplegrad.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE2]], target("spirv.Sampler") %[[SAMPLER_H2]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x float> %[[DDX_VAL]], <[[DIM]] x float> %[[DDY_VAL]], <[[DIM]] x i32> %[[OFFSET_VAL]])

// CHECK: @test_clamp(float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]])
// CHECK: %[[CALL_CLAMP:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]], int vector[[[DIM]]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x i32> noundef [[OFFSET_CONST]], float {{.*}} 1.000000e+00)
// CHECK: ret <4 x float> %[[CALL_CLAMP]]

float4 test_clamp(COORD_TYPE loc : LOC, GRAD_TYPE ddx : DDX, GRAD_TYPE ddy : DDY) : SV_Target {
  return t.SampleGrad(s, loc, ddx, ddy, OFFSET_ARG, 1.0f);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::SampleGrad(hlsl::SamplerState, float vector[[[COORD_DIM]]], float vector[[[DIM]]], float vector[[[DIM]]], int vector[[[DIM]]], float)(
// CHECK-SAME: ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^,]+]], <[[DIM]] x float> {{.*}} %[[DDX:[^,]+]], <[[DIM]] x float> {{.*}} %[[DDY:[^,]+]], <[[DIM]] x i32> {{.*}} %[[OFFSET:[^,]+]], float {{.*}} %[[CLAMP:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: %[[DDX_ADDR:.*]] = alloca <[[DIM]] x float>
// CHECK: %[[DDY_ADDR:.*]] = alloca <[[DIM]] x float>
// CHECK: %[[OFFSET_ADDR:.*]] = alloca <[[DIM]] x i32>
// CHECK: %[[CLAMP_ADDR:.*]] = alloca float
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: store <[[DIM]] x float> %[[DDX]], ptr %[[DDX_ADDR]]
// CHECK: store <[[DIM]] x float> %[[DDY]], ptr %[[DDY_ADDR]]
// CHECK: store <[[DIM]] x i32> %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK: store float %[[CLAMP]], ptr %[[CLAMP_ADDR]]
// CHECK: %[[THIS_VAL3:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL3]], i32 0, i32 0
// CHECK: %[[HANDLE3:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP3]]
// CHECK: %[[SAMPLER_GEP3:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H3:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP3]]
// CHECK: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// CHECK: %[[DDX_VAL:.*]] = load <[[DIM]] x float>, ptr %[[DDX_ADDR]]
// CHECK: %[[DDY_VAL:.*]] = load <[[DIM]] x float>, ptr %[[DDY_ADDR]]
// CHECK: %[[OFFSET_VAL:.*]] = load <[[DIM]] x i32>, ptr %[[OFFSET_ADDR]]
// CHECK: %[[CLAMP_VAL:.*]] = load float, ptr %[[CLAMP_ADDR]]
// CHECK: %[[CLAMP_CAST3:.*]] = fptrunc {{.*}} double {{.*}} to float
// DXIL: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.samplegrad.clamp.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE3]], target("dx.Sampler", 0) %[[SAMPLER_H3]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x float> %[[DDX_VAL]], <[[DIM]] x float> %[[DDY_VAL]], <[[DIM]] x i32> %[[OFFSET_VAL]], float %[[CLAMP_CAST3]])
// SPIRV: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.samplegrad.clamp.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE3]], target("spirv.Sampler") %[[SAMPLER_H3]], <[[COORD_DIM]] x float> %[[COORD_VAL]], <[[DIM]] x float> %[[DDX_VAL]], <[[DIM]] x float> %[[DDY_VAL]], <[[DIM]] x i32> %[[OFFSET_VAL]], float %[[CLAMP_CAST3]])
