// Texture2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2D -DLOCATION_TYPE=int3 -DZEROS=0 -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=Texture2D -DLOCATION_DIM=3 -DCOORD_DIM=2 -DCOORD_MASK="<i32 0, i32 1>" -DDXIL_TY=2 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2D -DLOCATION_TYPE=int3 -DZEROS=0 -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=Texture2D -DLOCATION_DIM=3 -DCOORD_DIM=2 -DCOORD_MASK="<i32 0, i32 1>" -DARRAYED=0 -DSAMPLED=1 -DFORMAT1=0 -DFORMAT3=0 -DFORMAT6=0 -DFORMAT21=0 -DFORMAT24=0 -DFORMAT25=0

// Texture2DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DArray -DLOCATION_TYPE=int4 -DZEROS=" 0, 0" -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=Texture2DArray -DLOCATION_DIM=4 -DCOORD_DIM=3 -DCOORD_MASK="<i32 0, i32 1, i32 2>" -DDXIL_TY=7 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DArray -DLOCATION_TYPE=int4 -DZEROS=" 0, 0" -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=Texture2DArray -DLOCATION_DIM=4 -DCOORD_DIM=3 -DCOORD_MASK="<i32 0, i32 1, i32 2>" -DARRAYED=1 -DSAMPLED=1 -DFORMAT1=0 -DFORMAT3=0 -DFORMAT6=0 -DFORMAT21=0 -DFORMAT24=0 -DFORMAT25=0

// RWTexture2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=RWTexture2D -DLOCATION_TYPE=int3 -DZEROS=0 -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=RWTexture2D -DLOCATION_DIM=3 -DCOORD_DIM=2 -DCOORD_MASK="<i32 0, i32 1>" -DDXIL_TY=2 -DRW=1
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=RWTexture2D -DLOCATION_TYPE=int3 -DZEROS=0 -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=RWTexture2D -DLOCATION_DIM=3 -DCOORD_DIM=2 -DCOORD_MASK="<i32 0, i32 1>" -DARRAYED=0 -DSAMPLED=2 -DFORMAT1=1 -DFORMAT3=3 -DFORMAT6=6 -DFORMAT21=21 -DFORMAT24=24 -DFORMAT25=25

// RWTexture2DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=RWTexture2DArray -DLOCATION_TYPE=int4 -DZEROS=" 0, 0" -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=RWTexture2DArray -DLOCATION_DIM=4 -DCOORD_DIM=3 -DCOORD_MASK="<i32 0, i32 1, i32 2>" -DDXIL_TY=7 -DRW=1
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=RWTexture2DArray -DLOCATION_TYPE=int4 -DZEROS=" 0, 0" -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=RWTexture2DArray -DLOCATION_DIM=4 -DCOORD_DIM=3 -DCOORD_MASK="<i32 0, i32 1, i32 2>" -DARRAYED=1 -DSAMPLED=2 -DFORMAT1=1 -DFORMAT3=3 -DFORMAT6=6 -DFORMAT21=21 -DFORMAT24=24 -DFORMAT25=25

TEXTURE<float4> t;

// CHECK: define hidden {{.*}} <4 x float> @test_load(int vector[2])
// CHECK: %[[COORD:.*]] = insertelement <[[LOCATION_DIM]] x i32> {{.*}}, i32 0, i32 [[COORD_DIM]]
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} @t, <[[LOCATION_DIM]] x i32> noundef %[[COORD]])
// CHECK: ret <4 x float> %[[CALL]]

float4 test_load(int2 loc : LOC) : SV_Target {
  return t.Load(LOCATION_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[LOCATION_ADDR:.*]] = alloca <[[LOCATION_DIM]] x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[LOCATION_DIM]] x i32> %[[LOCATION]], ptr %[[LOCATION_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("{{(dx.Texture|spirv.Image)}}", {{.*}}), ptr %[[HANDLE_GEP]]
// CHECK: %[[LOCATION_VAL:.*]] = load <[[LOCATION_DIM]] x i32>, ptr %[[LOCATION_ADDR]]
// CHECK: %[[COORD:.*]] = shufflevector <[[LOCATION_DIM]] x i32> %[[LOCATION_VAL]], <[[LOCATION_DIM]] x i32> poison, <[[COORD_DIM]] x i32> [[COORD_MASK]]
// CHECK: %[[LOD:.*]] = extractelement <[[LOCATION_DIM]] x i32> %[[LOCATION_VAL]], i64 [[COORD_DIM]]
// DXIL: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[COORD]], i32 %[[LOD]], <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT1]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[COORD]], i32 %[[LOD]], <2 x i32> zeroinitializer)
// CHECK: ret <4 x float> %[[RES]]

// CHECK: define hidden {{.*}} <4 x float> @test_load_offset(int vector[2])
// CHECK: %[[COORD:.*]] = insertelement <[[LOCATION_DIM]] x i32> {{.*}}, i32 0, i32 [[COORD_DIM]]
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} @t, <[[LOCATION_DIM]] x i32> noundef %[[COORD]], <2 x i32> noundef splat (i32 1))
// CHECK: ret <4 x float> %[[CALL]]

float4 test_load_offset(int2 loc : LOC) : SV_Target {
  return t.Load(LOCATION_TYPE(loc, ZEROS), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[LOCATION_ADDR:.*]] = alloca <[[LOCATION_DIM]] x i32>
// CHECK: %[[OFFSET_ADDR:.*]] = alloca <2 x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[LOCATION_DIM]] x i32> %[[LOCATION]], ptr %[[LOCATION_ADDR]]
// CHECK: store <2 x i32> %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("{{(dx.Texture|spirv.Image)}}", {{.*}}), ptr %[[HANDLE_GEP]]
// CHECK: %[[LOCATION_VAL:.*]] = load <[[LOCATION_DIM]] x i32>, ptr %[[LOCATION_ADDR]]
// CHECK: %[[COORD:.*]] = shufflevector <[[LOCATION_DIM]] x i32> %[[LOCATION_VAL]], <[[LOCATION_DIM]] x i32> poison, <[[COORD_DIM]] x i32> [[COORD_MASK]]
// CHECK: %[[LOD:.*]] = extractelement <[[LOCATION_DIM]] x i32> %[[LOCATION_VAL]], i64 [[COORD_DIM]]
// CHECK: %[[OFFSET_VAL:.*]] = load <2 x i32>, ptr %[[OFFSET_ADDR]]
// DXIL: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_{{.*}}("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[COORD]], i32 %[[LOD]], <2 x i32> %[[OFFSET_VAL]])
// SPIRV: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT1]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[COORD]], i32 %[[LOD]], <2 x i32> %[[OFFSET_VAL]])
// CHECK: ret <4 x float> %[[RES]]


// For the rest of the types, we just check that the call to the member
// function has the correct return type.

TEXTURE<float> t_float;

// CHECK: define hidden {{.*}} float @test_load_float(int vector[2])
// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call {{.*}} float @llvm.dx.resource.load.level.f32.tdx.Texture_f32_{{.*}}("dx.Texture", float, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call {{.*}} float @llvm.spv.resource.load.level.f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT3]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret float %[[RES]]
float test_load_float(int2 loc : LOC) {
  return t_float.Load(LOCATION_TYPE(loc, ZEROS));
}

// CHECK: define hidden {{.*}} float @test_load_offset_float(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} @t_float, <[[LOCATION_DIM]] x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret float %[[CALL]]
float test_load_offset_float(int2 loc : LOC) {
  return t_float.Load(LOCATION_TYPE(loc, ZEROS), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call {{.*}} float @llvm.dx.resource.load.level.f32.tdx.Texture_f32_{{.*}}("dx.Texture", float, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call {{.*}} float @llvm.spv.resource.load.level.f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT3]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret float %[[RES]]

TEXTURE<float2> t_float2;

// CHECK: define hidden {{.*}} <2 x float> @test_load_float2(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <2 x float> @hlsl::[[TEXTURE]]<float vector[2]>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} @t_float2, <[[LOCATION_DIM]] x i32> noundef %{{.*}})
// CHECK: ret <2 x float> %[[CALL]]
float2 test_load_float2(int2 loc : LOC) {
  return t_float2.Load(LOCATION_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} <2 x float> @hlsl::[[TEXTURE]]<float vector[2]>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call {{.*}} <2 x float> @llvm.dx.resource.load.level.v2f32.tdx.Texture_v2f32_{{.*}}("dx.Texture", <2 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call {{.*}} <2 x float> @llvm.spv.resource.load.level.v2f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT6]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret <2 x float> %[[RES]]

// CHECK: define hidden {{.*}} <2 x float> @test_load_offset_float2(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <2 x float> @hlsl::[[TEXTURE]]<float vector[2]>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} @t_float2, <[[LOCATION_DIM]] x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret <2 x float> %[[CALL]]
float2 test_load_offset_float2(int2 loc : LOC) {
  return t_float2.Load(LOCATION_TYPE(loc, ZEROS), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} <2 x float> @hlsl::[[TEXTURE]]<float vector[2]>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call {{.*}} <2 x float> @llvm.dx.resource.load.level.v2f32.tdx.Texture_v2f32_{{.*}}("dx.Texture", <2 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call {{.*}} <2 x float> @llvm.spv.resource.load.level.v2f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT6]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret <2 x float> %[[RES]]

TEXTURE<float3> t_float3;

// CHECK: define hidden {{.*}} <3 x float> @test_load_float3(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x float> @hlsl::[[TEXTURE]]<float vector[3]>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} @t_float3, <[[LOCATION_DIM]] x i32> noundef %{{.*}})
// CHECK: ret <3 x float> %[[CALL]]
float3 test_load_float3(int2 loc : LOC) {
  return t_float3.Load(LOCATION_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x float> @hlsl::[[TEXTURE]]<float vector[3]>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call {{.*}} <3 x float> @llvm.dx.resource.load.level.v3f32.tdx.Texture_v3f32_{{.*}}("dx.Texture", <3 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call {{.*}} <3 x float> @llvm.spv.resource.load.level.v3f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret <3 x float> %[[RES]]

// CHECK: define hidden {{.*}} <3 x float> @test_load_offset_float3(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x float> @hlsl::[[TEXTURE]]<float vector[3]>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} @t_float3, <[[LOCATION_DIM]] x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret <3 x float> %[[CALL]]
float3 test_load_offset_float3(int2 loc : LOC) {
  return t_float3.Load(LOCATION_TYPE(loc, ZEROS), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x float> @hlsl::[[TEXTURE]]<float vector[3]>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call {{.*}} <3 x float> @llvm.dx.resource.load.level.v3f32.tdx.Texture_v3f32_{{.*}}("dx.Texture", <3 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call {{.*}} <3 x float> @llvm.spv.resource.load.level.v3f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret <3 x float> %[[RES]]

TEXTURE<int> t_int;

// CHECK: define hidden {{.*}} i32 @test_load_int(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} i32 @hlsl::[[TEXTURE]]<int>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} @t_int, <[[LOCATION_DIM]] x i32> noundef %{{.*}})
// CHECK: ret i32 %[[CALL]]
int test_load_int(int2 loc : LOC) {
  return t_int.Load(LOCATION_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} i32 @hlsl::[[TEXTURE]]<int>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call i32 @llvm.dx.resource.load.level.i32.tdx.Texture_i32_{{.*}}("dx.Texture", i32, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call i32 @llvm.spv.resource.load.level.i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT24]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret i32 %[[RES]]

// CHECK: define hidden {{.*}} i32 @test_load_offset_int(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} i32 @hlsl::[[TEXTURE]]<int>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} @t_int, <[[LOCATION_DIM]] x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret i32 %[[CALL]]
int test_load_offset_int(int2 loc : LOC) {
  return t_int.Load(LOCATION_TYPE(loc, ZEROS), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} i32 @hlsl::[[TEXTURE]]<int>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call i32 @llvm.dx.resource.load.level.i32.tdx.Texture_i32_{{.*}}("dx.Texture", i32, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call i32 @llvm.spv.resource.load.level.i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT24]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret i32 %[[RES]]

TEXTURE<int2> t_int2;

// CHECK: define hidden {{.*}} <2 x i32> @test_load_int2(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <2 x i32> @hlsl::[[TEXTURE]]<int vector[2]>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} @t_int2, <[[LOCATION_DIM]] x i32> noundef %{{.*}})
// CHECK: ret <2 x i32> %[[CALL]]
int2 test_load_int2(int2 loc : LOC) {
  return t_int2.Load(LOCATION_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} <2 x i32> @hlsl::[[TEXTURE]]<int vector[2]>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call <2 x i32> @llvm.dx.resource.load.level.v2i32.tdx.Texture_v2i32_{{.*}}("dx.Texture", <2 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call <2 x i32> @llvm.spv.resource.load.level.v2i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT25]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret <2 x i32> %[[RES]]

// CHECK: define hidden {{.*}} <2 x i32> @test_load_offset_int2(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <2 x i32> @hlsl::[[TEXTURE]]<int vector[2]>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} @t_int2, <[[LOCATION_DIM]] x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret <2 x i32> %[[CALL]]
int2 test_load_offset_int2(int2 loc : LOC) {
  return t_int2.Load(LOCATION_TYPE(loc, ZEROS), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} <2 x i32> @hlsl::[[TEXTURE]]<int vector[2]>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call <2 x i32> @llvm.dx.resource.load.level.v2i32.tdx.Texture_v2i32_{{.*}}("dx.Texture", <2 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call <2 x i32> @llvm.spv.resource.load.level.v2i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT25]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret <2 x i32> %[[RES]]

TEXTURE<int3> t_int3;

// CHECK: define hidden {{.*}} <3 x i32> @test_load_int3(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x i32> @hlsl::[[TEXTURE]]<int vector[3]>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} @t_int3, <[[LOCATION_DIM]] x i32> noundef %{{.*}})
// CHECK: ret <3 x i32> %[[CALL]]
int3 test_load_int3(int2 loc : LOC) {
  return t_int3.Load(LOCATION_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x i32> @hlsl::[[TEXTURE]]<int vector[3]>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.load.level.v3i32.tdx.Texture_v3i32_{{.*}}("dx.Texture", <3 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.load.level.v3i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret <3 x i32> %[[RES]]

// CHECK: define hidden {{.*}} <3 x i32> @test_load_offset_int3(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x i32> @hlsl::[[TEXTURE]]<int vector[3]>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} @t_int3, <[[LOCATION_DIM]] x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret <3 x i32> %[[CALL]]
int3 test_load_offset_int3(int2 loc : LOC) {
  return t_int3.Load(LOCATION_TYPE(loc, ZEROS), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x i32> @hlsl::[[TEXTURE]]<int vector[3]>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.load.level.v3i32.tdx.Texture_v3i32_{{.*}}("dx.Texture", <3 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.load.level.v3i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret <3 x i32> %[[RES]]

TEXTURE<int4> t_int4;

// CHECK: define hidden {{.*}} <4 x i32> @test_load_int4(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x i32> @hlsl::[[TEXTURE]]<int vector[4]>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} @t_int4, <[[LOCATION_DIM]] x i32> noundef %{{.*}})
// CHECK: ret <4 x i32> %[[CALL]]
int4 test_load_int4(int2 loc : LOC) {
  return t_int4.Load(LOCATION_TYPE(loc, ZEROS));
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x i32> @hlsl::[[TEXTURE]]<int vector[4]>::Load(int vector[[[LOCATION_DIM]]])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call <4 x i32> @llvm.dx.resource.load.level.v4i32.tdx.Texture_v4i32_{{.*}}("dx.Texture", <4 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call <4 x i32> @llvm.spv.resource.load.level.v4i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT21]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret <4 x i32> %[[RES]]

// CHECK: define hidden {{.*}} <4 x i32> @test_load_offset_int4(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x i32> @hlsl::[[TEXTURE]]<int vector[4]>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} @t_int4, <[[LOCATION_DIM]] x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret <4 x i32> %[[CALL]]
int4 test_load_offset_int4(int2 loc : LOC) {
  return t_int4.Load(LOCATION_TYPE(loc, ZEROS), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x i32> @hlsl::[[TEXTURE]]<int vector[4]>::Load(int vector[[[LOCATION_DIM]]], int vector[2])(ptr {{.*}} %[[THIS:.*]], <[[LOCATION_DIM]] x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call <4 x i32> @llvm.dx.resource.load.level.v4i32.tdx.Texture_v4i32_{{.*}}("dx.Texture", <4 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call <4 x i32> @llvm.spv.resource.load.level.v4i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT21]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret <4 x i32> %[[RES]]
