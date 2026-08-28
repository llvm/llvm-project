// Texture2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DOFFSET_ARG="int2(1, 1)" -DTEXTURE=Texture2D -DLOAD_TYPE=int3 \
// RUN:   -DZEROS=0 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=Texture2D \
// RUN:   -DLOAD_DIM=3 -DCOORD_DIM=2 -DCOORD_MASK="<i32 0, i32 1>" -DDXIL_TY=2 \
// RUN:   -DRW=0 -DENTRY_DIM=2 -DDIM=2
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DOFFSET_ARG="int2(1, 1)" -DTEXTURE=Texture2D -DLOAD_TYPE=int3 \
// RUN:   -DZEROS=0 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=Texture2D \
// RUN:   -DLOAD_DIM=3 -DCOORD_DIM=2 -DCOORD_MASK="<i32 0, i32 1>" -DARRAYED=0 \
// RUN:   -DSAMPLED=1 -DFORMAT1=0 -DFORMAT3=0 -DFORMAT6=0 -DFORMAT21=0 \
// RUN:   -DFORMAT24=0 -DFORMAT25=0 -DSPV_DIM=1 -DENTRY_DIM=2 -DDIM=2

// Texture2DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DOFFSET_ARG="int2(1, 1)" -DTEXTURE=Texture2DArray -DLOAD_TYPE=int4 \
// RUN:   -DZEROS=" 0, 0" -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=Texture2DArray \
// RUN:   -DLOAD_DIM=4 -DCOORD_DIM=3 -DCOORD_MASK="<i32 0, i32 1, i32 2>" \
// RUN:   -DDXIL_TY=7 -DRW=0 -DENTRY_DIM=2 -DDIM=2
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DOFFSET_ARG="int2(1, 1)" -DTEXTURE=Texture2DArray -DLOAD_TYPE=int4 \
// RUN:   -DZEROS=" 0, 0" -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=Texture2DArray \
// RUN:   -DLOAD_DIM=4 -DCOORD_DIM=3 -DCOORD_MASK="<i32 0, i32 1, i32 2>" \
// RUN:   -DARRAYED=1 -DSAMPLED=1 -DFORMAT1=0 -DFORMAT3=0 -DFORMAT6=0 \
// RUN:   -DFORMAT21=0 -DFORMAT24=0 -DFORMAT25=0 -DSPV_DIM=1 -DENTRY_DIM=2 \
// RUN:   -DDIM=2

// RWTexture2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DOFFSET_ARG="int2(1, 1)" -DTEXTURE=RWTexture2D -DLOAD_TYPE=int3 \
// RUN:   -DZEROS=0 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=RWTexture2D \
// RUN:   -DLOAD_DIM=3 -DCOORD_DIM=2 -DCOORD_MASK="<i32 0, i32 1>" -DDXIL_TY=2 \
// RUN:   -DRW=1 -DENTRY_DIM=2 -DDIM=2
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DOFFSET_ARG="int2(1, 1)" -DTEXTURE=RWTexture2D -DLOAD_TYPE=int3 \
// RUN:   -DZEROS=0 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=RWTexture2D \
// RUN:   -DLOAD_DIM=3 -DCOORD_DIM=2 -DCOORD_MASK="<i32 0, i32 1>" -DARRAYED=0 \
// RUN:   -DSAMPLED=2 -DFORMAT1=1 -DFORMAT3=3 -DFORMAT6=6 -DFORMAT21=21 \
// RUN:   -DFORMAT24=24 -DFORMAT25=25 -DSPV_DIM=1 -DENTRY_DIM=2 -DDIM=2

// RWTexture2DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DOFFSET_ARG="int2(1, 1)" -DTEXTURE=RWTexture2DArray \
// RUN:   -DLOAD_TYPE=int4 -DZEROS=" 0, 0" -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,DXIL \
// RUN:   -DTEXTURE=RWTexture2DArray -DLOAD_DIM=4 -DCOORD_DIM=3 \
// RUN:   -DCOORD_MASK="<i32 0, i32 1, i32 2>" -DDXIL_TY=7 -DRW=1 \
// RUN:   -DENTRY_DIM=2 -DDIM=2
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DOFFSET_ARG="int2(1, 1)" -DTEXTURE=RWTexture2DArray \
// RUN:   -DLOAD_TYPE=int4 -DZEROS=" 0, 0" -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SPIRV \
// RUN:   -DTEXTURE=RWTexture2DArray -DLOAD_DIM=4 -DCOORD_DIM=3 \
// RUN:   -DCOORD_MASK="<i32 0, i32 1, i32 2>" -DARRAYED=1 -DSAMPLED=2 \
// RUN:   -DFORMAT1=1 -DFORMAT3=3 -DFORMAT6=6 -DFORMAT21=21 -DFORMAT24=24 \
// RUN:   -DFORMAT25=25 -DSPV_DIM=1 -DENTRY_DIM=2 -DDIM=2

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   ENTRY_TYPE         the entry point's own coordinate type
//   OFFSET_ARG         a literal offset argument
//   TEXTURE            resource type name
//   LOAD_TYPE          Load location type
//   ZEROS              the trailing components padding ENTRY_TYPE out to
//                      LOAD_TYPE
//   LOAD_DIM           Load location components (COORD_DIM plus the mip level)
//   COORD_DIM          sample location components (DIM plus the array slice)
//   COORD_MASK         shufflevector mask extracting the coordinate from a
//                      location
//   DXIL_TY            dx.Texture resource-kind operand
//   RW                 dx.Texture UAV operand
//   ENTRY_DIM          the entry point's own coordinate components
//   DIM                number of resource dimensions (offset, ddx/ddy, LOD
//                      location)
//   ARRAYED            spirv.Image Arrayed operand
//   SAMPLED            spirv.Image Sampled operand
//   SPV_DIM            spirv.Image Dim operand
//   FORMAT<n>          spirv.Image Image Format operand for the element
//                      type of the texture declared on line <n>

TEXTURE<float4> t;

// CHECK: define hidden {{.*}} <4 x float> @test_load(int vector[[[ENTRY_DIM]]])
// CHECK: %[[COORD:.*]] = insertelement <[[LOAD_DIM]] x i32> {{.*}}, i32 0, i32 [[COORD_DIM]]
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} @t, <[[LOAD_DIM]] x i32> noundef %[[COORD]])
// CHECK: ret <4 x float> %[[CALL]]

float4 test_load(ENTRY_TYPE loc : LOC) : SV_Target {
  return t.Load(LOAD_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[LOAD_ADDR:.*]] = alloca <[[LOAD_DIM]] x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[LOAD_DIM]] x i32> %[[LOAD]], ptr %[[LOAD_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("{{(dx.Texture|spirv.Image)}}", {{.*}}), ptr %[[HANDLE_GEP]]
// CHECK: %[[LOAD_VAL:.*]] = load <[[LOAD_DIM]] x i32>, ptr %[[LOAD_ADDR]]
// CHECK: %[[COORD:.*]] = shufflevector <[[LOAD_DIM]] x i32> %[[LOAD_VAL]], <[[LOAD_DIM]] x i32> poison, <[[COORD_DIM]] x i32> [[COORD_MASK]]
// CHECK: %[[LOD:.*]] = extractelement <[[LOAD_DIM]] x i32> %[[LOAD_VAL]], i64 [[COORD_DIM]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[COORD]], i32 %[[LOD]], <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT1]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[COORD]], i32 %[[LOD]], <[[DIM]] x i32> zeroinitializer)
// CHECK: ret <4 x float> %[[RES]]

// CHECK: define hidden {{.*}} <4 x float> @test_load_offset(int vector[[[ENTRY_DIM]]])
// CHECK: %[[COORD:.*]] = insertelement <[[LOAD_DIM]] x i32> {{.*}}, i32 0, i32 [[COORD_DIM]]
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} @t, <[[LOAD_DIM]] x i32> noundef %[[COORD]], <[[DIM]] x i32> noundef splat (i32 1))
// CHECK: ret <4 x float> %[[CALL]]

float4 test_load_offset(ENTRY_TYPE loc : LOC) : SV_Target {
  return t.Load(LOAD_TYPE(loc, ZEROS), OFFSET_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]], <[[DIM]] x i32> {{.*}} %[[OFFSET:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[LOAD_ADDR:.*]] = alloca <[[LOAD_DIM]] x i32>
// CHECK: %[[OFFSET_ADDR:.*]] = alloca <[[DIM]] x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[LOAD_DIM]] x i32> %[[LOAD]], ptr %[[LOAD_ADDR]]
// CHECK: store <[[DIM]] x i32> %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("{{(dx.Texture|spirv.Image)}}", {{.*}}), ptr %[[HANDLE_GEP]]
// CHECK: %[[LOAD_VAL:.*]] = load <[[LOAD_DIM]] x i32>, ptr %[[LOAD_ADDR]]
// CHECK: %[[COORD:.*]] = shufflevector <[[LOAD_DIM]] x i32> %[[LOAD_VAL]], <[[LOAD_DIM]] x i32> poison, <[[COORD_DIM]] x i32> [[COORD_MASK]]
// CHECK: %[[LOD:.*]] = extractelement <[[LOAD_DIM]] x i32> %[[LOAD_VAL]], i64 [[COORD_DIM]]
// CHECK: %[[OFFSET_VAL:.*]] = load <[[DIM]] x i32>, ptr %[[OFFSET_ADDR]]
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_{{.*}}("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[COORD]], i32 %[[LOD]], <[[DIM]] x i32> %[[OFFSET_VAL]])
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT1]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[COORD]], i32 %[[LOD]], <[[DIM]] x i32> %[[OFFSET_VAL]])
// CHECK: ret <4 x float> %[[RES]]


// For the rest of the types, we just check that the call to the member
// function has the correct return type.

TEXTURE<float> t_float;

// CHECK: define hidden {{.*}} float @test_load_float(int vector[[[ENTRY_DIM]]])
// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.load.level.f32.tdx.Texture_f32_{{.*}}("dx.Texture", float, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.load.level.f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT3]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// CHECK: ret float %[[RES]]
float test_load_float(ENTRY_TYPE loc : LOC) {
  return t_float.Load(LOAD_TYPE(loc, ZEROS));
}

// CHECK: define hidden {{.*}} float @test_load_offset_float(int vector[[[ENTRY_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} @t_float, <[[LOAD_DIM]] x i32> noundef %{{.*}}, <[[DIM]] x i32> noundef splat (i32 1))
// CHECK: ret float %[[CALL]]
float test_load_offset_float(ENTRY_TYPE loc : LOC) {
  return t_float.Load(LOAD_TYPE(loc, ZEROS), OFFSET_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]], <[[DIM]] x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.load.level.f32.tdx.Texture_f32_{{.*}}("dx.Texture", float, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.load.level.f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT3]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// CHECK: ret float %[[RES]]

TEXTURE<float2> t_float2;

// CHECK: define hidden {{.*}} <2 x float> @test_load_float2(int vector[[[ENTRY_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <2 x float> @hlsl::[[TEXTURE]]<float vector[2]>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} @t_float2, <[[LOAD_DIM]] x i32> noundef %{{.*}})
// CHECK: ret <2 x float> %[[CALL]]
float2 test_load_float2(ENTRY_TYPE loc : LOC) {
  return t_float2.Load(LOAD_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} <2 x float> @hlsl::[[TEXTURE]]<float vector[2]>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.dx.resource.load.level.v2f32.tdx.Texture_v2f32_{{.*}}("dx.Texture", <2 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.resource.load.level.v2f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT6]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// CHECK: ret <2 x float> %[[RES]]

// CHECK: define hidden {{.*}} <2 x float> @test_load_offset_float2(int vector[[[ENTRY_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <2 x float> @hlsl::[[TEXTURE]]<float vector[2]>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} @t_float2, <[[LOAD_DIM]] x i32> noundef %{{.*}}, <[[DIM]] x i32> noundef splat (i32 1))
// CHECK: ret <2 x float> %[[CALL]]
float2 test_load_offset_float2(ENTRY_TYPE loc : LOC) {
  return t_float2.Load(LOAD_TYPE(loc, ZEROS), OFFSET_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <2 x float> @hlsl::[[TEXTURE]]<float vector[2]>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]], <[[DIM]] x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.dx.resource.load.level.v2f32.tdx.Texture_v2f32_{{.*}}("dx.Texture", <2 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.resource.load.level.v2f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT6]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// CHECK: ret <2 x float> %[[RES]]

TEXTURE<float3> t_float3;

// CHECK: define hidden {{.*}} <3 x float> @test_load_float3(int vector[[[ENTRY_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x float> @hlsl::[[TEXTURE]]<float vector[3]>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} @t_float3, <[[LOAD_DIM]] x i32> noundef %{{.*}})
// CHECK: ret <3 x float> %[[CALL]]
float3 test_load_float3(ENTRY_TYPE loc : LOC) {
  return t_float3.Load(LOAD_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x float> @hlsl::[[TEXTURE]]<float vector[3]>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.dx.resource.load.level.v3f32.tdx.Texture_v3f32_{{.*}}("dx.Texture", <3 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.spv.resource.load.level.v3f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// CHECK: ret <3 x float> %[[RES]]

// CHECK: define hidden {{.*}} <3 x float> @test_load_offset_float3(int vector[[[ENTRY_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x float> @hlsl::[[TEXTURE]]<float vector[3]>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} @t_float3, <[[LOAD_DIM]] x i32> noundef %{{.*}}, <[[DIM]] x i32> noundef splat (i32 1))
// CHECK: ret <3 x float> %[[CALL]]
float3 test_load_offset_float3(ENTRY_TYPE loc : LOC) {
  return t_float3.Load(LOAD_TYPE(loc, ZEROS), OFFSET_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x float> @hlsl::[[TEXTURE]]<float vector[3]>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]], <[[DIM]] x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.dx.resource.load.level.v3f32.tdx.Texture_v3f32_{{.*}}("dx.Texture", <3 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.spv.resource.load.level.v3f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// CHECK: ret <3 x float> %[[RES]]

TEXTURE<int> t_int;

// CHECK: define hidden {{.*}} i32 @test_load_int(int vector[[[ENTRY_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} i32 @hlsl::[[TEXTURE]]<int>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} @t_int, <[[LOAD_DIM]] x i32> noundef %{{.*}})
// CHECK: ret i32 %[[CALL]]
int test_load_int(ENTRY_TYPE loc : LOC) {
  return t_int.Load(LOAD_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} i32 @hlsl::[[TEXTURE]]<int>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call i32 @llvm.dx.resource.load.level.i32.tdx.Texture_i32_{{.*}}("dx.Texture", i32, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call i32 @llvm.spv.resource.load.level.i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT24]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// CHECK: ret i32 %[[RES]]

// CHECK: define hidden {{.*}} i32 @test_load_offset_int(int vector[[[ENTRY_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} i32 @hlsl::[[TEXTURE]]<int>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} @t_int, <[[LOAD_DIM]] x i32> noundef %{{.*}}, <[[DIM]] x i32> noundef splat (i32 1))
// CHECK: ret i32 %[[CALL]]
int test_load_offset_int(ENTRY_TYPE loc : LOC) {
  return t_int.Load(LOAD_TYPE(loc, ZEROS), OFFSET_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} i32 @hlsl::[[TEXTURE]]<int>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]], <[[DIM]] x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call i32 @llvm.dx.resource.load.level.i32.tdx.Texture_i32_{{.*}}("dx.Texture", i32, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call i32 @llvm.spv.resource.load.level.i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT24]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// CHECK: ret i32 %[[RES]]

TEXTURE<int2> t_int2;

// CHECK: define hidden {{.*}} <[[DIM]] x i32> @test_load_int2(int vector[[[ENTRY_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <[[DIM]] x i32> @hlsl::[[TEXTURE]]<int vector[[[DIM]]]>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} @t_int2, <[[LOAD_DIM]] x i32> noundef %{{.*}})
// CHECK: ret <[[DIM]] x i32> %[[CALL]]
int2 test_load_int2(ENTRY_TYPE loc : LOC) {
  return t_int2.Load(LOAD_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} <[[DIM]] x i32> @hlsl::[[TEXTURE]]<int vector[[[DIM]]]>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call <[[DIM]] x i32> @llvm.dx.resource.load.level.v2i32.tdx.Texture_v2i32_{{.*}}("dx.Texture", <[[DIM]] x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call <[[DIM]] x i32> @llvm.spv.resource.load.level.v2i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT25]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// CHECK: ret <[[DIM]] x i32> %[[RES]]

// CHECK: define hidden {{.*}} <[[DIM]] x i32> @test_load_offset_int2(int vector[[[ENTRY_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <[[DIM]] x i32> @hlsl::[[TEXTURE]]<int vector[[[DIM]]]>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} @t_int2, <[[LOAD_DIM]] x i32> noundef %{{.*}}, <[[DIM]] x i32> noundef splat (i32 1))
// CHECK: ret <[[DIM]] x i32> %[[CALL]]
int2 test_load_offset_int2(ENTRY_TYPE loc : LOC) {
  return t_int2.Load(LOAD_TYPE(loc, ZEROS), OFFSET_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <[[DIM]] x i32> @hlsl::[[TEXTURE]]<int vector[[[DIM]]]>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]], <[[DIM]] x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call <[[DIM]] x i32> @llvm.dx.resource.load.level.v2i32.tdx.Texture_v2i32_{{.*}}("dx.Texture", <[[DIM]] x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call <[[DIM]] x i32> @llvm.spv.resource.load.level.v2i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT25]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// CHECK: ret <[[DIM]] x i32> %[[RES]]

TEXTURE<int3> t_int3;

// CHECK: define hidden {{.*}} <3 x i32> @test_load_int3(int vector[[[ENTRY_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x i32> @hlsl::[[TEXTURE]]<int vector[3]>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} @t_int3, <[[LOAD_DIM]] x i32> noundef %{{.*}})
// CHECK: ret <3 x i32> %[[CALL]]
int3 test_load_int3(ENTRY_TYPE loc : LOC) {
  return t_int3.Load(LOAD_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x i32> @hlsl::[[TEXTURE]]<int vector[3]>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.load.level.v3i32.tdx.Texture_v3i32_{{.*}}("dx.Texture", <3 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.load.level.v3i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// CHECK: ret <3 x i32> %[[RES]]

// CHECK: define hidden {{.*}} <3 x i32> @test_load_offset_int3(int vector[[[ENTRY_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x i32> @hlsl::[[TEXTURE]]<int vector[3]>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} @t_int3, <[[LOAD_DIM]] x i32> noundef %{{.*}}, <[[DIM]] x i32> noundef splat (i32 1))
// CHECK: ret <3 x i32> %[[CALL]]
int3 test_load_offset_int3(ENTRY_TYPE loc : LOC) {
  return t_int3.Load(LOAD_TYPE(loc, ZEROS), OFFSET_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x i32> @hlsl::[[TEXTURE]]<int vector[3]>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]], <[[DIM]] x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.load.level.v3i32.tdx.Texture_v3i32_{{.*}}("dx.Texture", <3 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.load.level.v3i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// CHECK: ret <3 x i32> %[[RES]]

TEXTURE<int4> t_int4;

// CHECK: define hidden {{.*}} <4 x i32> @test_load_int4(int vector[[[ENTRY_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x i32> @hlsl::[[TEXTURE]]<int vector[4]>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} @t_int4, <[[LOAD_DIM]] x i32> noundef %{{.*}})
// CHECK: ret <4 x i32> %[[CALL]]
int4 test_load_int4(ENTRY_TYPE loc : LOC) {
  return t_int4.Load(LOAD_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x i32> @hlsl::[[TEXTURE]]<int vector[4]>::Load(int vector[[[LOAD_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call <4 x i32> @llvm.dx.resource.load.level.v4i32.tdx.Texture_v4i32_{{.*}}("dx.Texture", <4 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call <4 x i32> @llvm.spv.resource.load.level.v4i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT21]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> zeroinitializer)
// CHECK: ret <4 x i32> %[[RES]]

// CHECK: define hidden {{.*}} <4 x i32> @test_load_offset_int4(int vector[[[ENTRY_DIM]]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x i32> @hlsl::[[TEXTURE]]<int vector[4]>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} @t_int4, <[[LOAD_DIM]] x i32> noundef %{{.*}}, <[[DIM]] x i32> noundef splat (i32 1))
// CHECK: ret <4 x i32> %[[CALL]]
int4 test_load_offset_int4(ENTRY_TYPE loc : LOC) {
  return t_int4.Load(LOAD_TYPE(loc, ZEROS), OFFSET_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x i32> @hlsl::[[TEXTURE]]<int vector[4]>::Load(int vector[[[LOAD_DIM]]], int vector[[[DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOAD_DIM]] x i32> {{.*}} %[[LOAD:.*]], <[[DIM]] x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call <4 x i32> @llvm.dx.resource.load.level.v4i32.tdx.Texture_v4i32_{{.*}}("dx.Texture", <4 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call <4 x i32> @llvm.spv.resource.load.level.v4i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT21]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <[[DIM]] x i32> %{{.*}})
// CHECK: ret <4 x i32> %[[RES]]
