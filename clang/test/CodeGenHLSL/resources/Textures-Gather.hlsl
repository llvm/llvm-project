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
// DXIL: %"class.hlsl::SamplerState" = type { target("dx.Sampler", 0) }
// DXIL: %"class.hlsl::SamplerComparisonState" = type { target("dx.Sampler", 0) }

// SPIRV: %"class.hlsl::[[TEXTURE]]" = type { target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// SPIRV: %"class.hlsl::SamplerState" = type { target("spirv.Sampler") }
// SPIRV: %"class.hlsl::SamplerComparisonState" = type { target("spirv.Sampler") }

TEXTURE<float4> t;
SamplerState s;
SamplerComparisonState sc;

// CHECK: define hidden {{.*}} <4 x float> @main(float vector[[[COORD_DIM]]])(<[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Gather(hlsl::SamplerState, float vector[[[COORD_DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}})
// CHECK: ret <4 x float> %[[CALL]]

float4 main(COORD_TYPE loc : LOC) : SV_Target {
  return t.Gather(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Gather(hlsl::SamplerState, float vector[[[COORD_DIM]]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.gather.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], i32 0, <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.gather.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], i32 0, <[[DIM]] x i32> zeroinitializer)
// CHECK: ret <4 x float> %[[RES]]

// CHECK: define hidden {{.*}} <4 x float> @test_offset(float vector[[[COORD_DIM]]])(<[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Gather(hlsl::SamplerState, float vector[[[COORD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, <[[DIM]] x i32> {{.*}} [[OFFSET_CONST]])
// CHECK: ret <4 x float> %[[CALL]]

float4 test_offset(COORD_TYPE loc : LOC) : SV_Target {
  return t.Gather(s, loc, OFFSET_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Gather(hlsl::SamplerState, float vector[[[COORD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^,]+]], <[[DIM]] x i32> {{.*}} %[[OFFSET:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: %[[OFFSET_ADDR:.*]] = alloca <[[DIM]] x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: store <[[DIM]] x i32> %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// CHECK: %[[OFFSET_VAL:.*]] = load <[[DIM]] x i32>, ptr %[[OFFSET_ADDR]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.gather.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], i32 0, <[[DIM]] x i32> %[[OFFSET_VAL]])
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.gather.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], i32 0, <[[DIM]] x i32> %[[OFFSET_VAL]])
// CHECK: ret <4 x float> %[[RES]]

// CHECK: define hidden {{.*}} <4 x float> @test_green(float vector[[[COORD_DIM]]])(<[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::GatherGreen(hlsl::SamplerState, float vector[[[COORD_DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}})
// CHECK: ret <4 x float> %[[CALL]]

float4 test_green(COORD_TYPE loc : LOC) : SV_Target {
  return t.GatherGreen(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::GatherGreen(hlsl::SamplerState, float vector[[[COORD_DIM]]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.gather.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], i32 1, <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.gather.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], i32 1, <[[DIM]] x i32> zeroinitializer)
// CHECK: ret <4 x float> %[[RES]]

// CHECK: define hidden {{.*}} <4 x float> @test_red(float vector[[[COORD_DIM]]])(<[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::GatherRed(hlsl::SamplerState, float vector[[[COORD_DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}})
// CHECK: ret <4 x float> %[[CALL]]

float4 test_red(COORD_TYPE loc : LOC) : SV_Target {
  return t.GatherRed(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::GatherRed(hlsl::SamplerState, float vector[[[COORD_DIM]]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.gather.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], i32 0, <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.gather.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], i32 0, <[[DIM]] x i32> zeroinitializer)
// CHECK: ret <4 x float> %[[RES]]

// CHECK: define hidden {{.*}} <4 x float> @test_blue(float vector[[[COORD_DIM]]])(<[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::GatherBlue(hlsl::SamplerState, float vector[[[COORD_DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}})
// CHECK: ret <4 x float> %[[CALL]]

float4 test_blue(COORD_TYPE loc : LOC) : SV_Target {
  return t.GatherBlue(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::GatherBlue(hlsl::SamplerState, float vector[[[COORD_DIM]]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.gather.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], i32 2, <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.gather.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], i32 2, <[[DIM]] x i32> zeroinitializer)
// CHECK: ret <4 x float> %[[RES]]

// CHECK: define hidden {{.*}} <4 x float> @test_alpha(float vector[[[COORD_DIM]]])(<[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::GatherAlpha(hlsl::SamplerState, float vector[[[COORD_DIM]]])(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}})
// CHECK: ret <4 x float> %[[CALL]]

float4 test_alpha(COORD_TYPE loc : LOC) : SV_Target {
  return t.GatherAlpha(s, loc);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::GatherAlpha(hlsl::SamplerState, float vector[[[COORD_DIM]]])(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.gather.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], i32 3, <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.gather.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], i32 3, <[[DIM]] x i32> zeroinitializer)
// CHECK: ret <4 x float> %[[RES]]

// CHECK: define hidden {{.*}} <4 x float> @test_cmp(float vector[[[COORD_DIM]]])(<[[COORD_DIM]] x float> noundef nofpclass(nan inf) %[[LOC:.*]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::GatherCmp(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float)(ptr {{.*}} @t, ptr {{.*}} byval(%"class.hlsl::SamplerComparisonState") {{.*}}, <[[COORD_DIM]] x float> {{.*}} %{{.*}}, float {{.*}} 5.000000e-01)
// CHECK: ret <4 x float> %[[CALL]]

float4 test_cmp(COORD_TYPE loc : LOC) : SV_Target {
  return t.GatherCmp(sc, loc, 0.5);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::GatherCmp(hlsl::SamplerComparisonState, float vector[[[COORD_DIM]]], float)(ptr {{.*}} %[[THIS:[^,]+]], ptr {{.*}} byval(%"class.hlsl::SamplerComparisonState") {{.*}} %[[SAMPLER:[^,]+]], <[[COORD_DIM]] x float> {{.*}} %[[COORD:[^,]+]], float {{.*}} %[[CMP:[^)]+]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[COORD_ADDR:.*]] = alloca <[[COORD_DIM]] x float>
// CHECK: %[[CMP_ADDR:.*]] = alloca float
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x float> %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: store float %[[CMP]], ptr %[[CMP_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target{{.*}}, ptr %[[HANDLE_GEP]]
// CHECK: %[[SAMPLER_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::SamplerComparisonState", ptr %[[SAMPLER]], i32 0, i32 0
// CHECK: %[[SAMPLER_H:.*]] = load target{{.*}}, ptr %[[SAMPLER_GEP]]
// CHECK: %[[COORD_VAL:.*]] = load <[[COORD_DIM]] x float>, ptr %[[COORD_ADDR]]
// CHECK: %[[CMP_VAL:.*]] = load float, ptr %[[CMP_ADDR]]
// CHECK: %[[CONV:.*]] = fpext {{.*}} float %[[CMP_VAL]] to double
// CHECK: %[[TRUNC:.*]] = fptrunc {{.*}} double %[[CONV]] to float
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.gather.cmp.v4f32.{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], target("dx.Sampler", 0) %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], float %[[TRUNC]], i32 0, <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.gather.cmp.v4f32.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], target("spirv.Sampler") %[[SAMPLER_H]], <[[COORD_DIM]] x float> %[[COORD_VAL]], float %[[TRUNC]], <[[DIM]] x i32> zeroinitializer)
// CHECK: ret <4 x float> %[[RES]]
