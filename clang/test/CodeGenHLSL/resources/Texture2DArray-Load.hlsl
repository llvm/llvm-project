// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV

Texture2DArray<float4> t;

// CHECK: define hidden {{.*}} <4 x float> @test_load(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::Load(int vector[4])(ptr {{.*}} @t, <4 x i32> noundef %{{.*}})
// CHECK: ret <4 x float> %[[CALL]]

float4 test_load(int2 loc : LOC) : SV_Target {
  return t.Load(int4(loc, 0, 0));
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::Load(int vector[4])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[LOCATION_ADDR:.*]] = alloca <4 x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <4 x i32> %[[LOCATION]], ptr %[[LOCATION_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2DArray", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("{{(dx.Texture|spirv.Image)}}", {{.*}}), ptr %[[HANDLE_GEP]]
// CHECK: %[[LOCATION_VAL:.*]] = load <4 x i32>, ptr %[[LOCATION_ADDR]]
// CHECK: %[[COORD:.*]] = shufflevector <4 x i32> %[[LOCATION_VAL]], <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK: %[[LOD:.*]] = extractelement <4 x i32> %[[LOCATION_VAL]], i64 3
// DXIL: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_7t.v3i32.i32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 7) %[[HANDLE]], <3 x i32> %[[COORD]], i32 %[[LOD]], <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %[[HANDLE]], <3 x i32> %[[COORD]], i32 %[[LOD]], <2 x i32> zeroinitializer)
// CHECK: ret <4 x float> %[[RES]]

// CHECK: define hidden {{.*}} <4 x float> @test_load_offset(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::Load(int vector[4], int vector[2])(ptr {{.*}} @t, <4 x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret <4 x float> %[[CALL]]

float4 test_load_offset(int2 loc : LOC) : SV_Target {
  return t.Load(int4(loc, 0, 0), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::Texture2DArray<float vector[4]>::Load(int vector[4], int vector[2])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[LOCATION_ADDR:.*]] = alloca <4 x i32>
// CHECK: %[[OFFSET_ADDR:.*]] = alloca <2 x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <4 x i32> %[[LOCATION]], ptr %[[LOCATION_ADDR]]
// CHECK: store <2 x i32> %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2DArray", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("{{(dx.Texture|spirv.Image)}}", {{.*}}), ptr %[[HANDLE_GEP]]
// CHECK: %[[LOCATION_VAL:.*]] = load <4 x i32>, ptr %[[LOCATION_ADDR]]
// CHECK: %[[COORD:.*]] = shufflevector <4 x i32> %[[LOCATION_VAL]], <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK: %[[LOD:.*]] = extractelement <4 x i32> %[[LOCATION_VAL]], i64 3
// CHECK: %[[OFFSET_VAL:.*]] = load <2 x i32>, ptr %[[OFFSET_ADDR]]
// DXIL: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_7t.v3i32.i32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 7) %[[HANDLE]], <3 x i32> %[[COORD]], i32 %[[LOD]], <2 x i32> %[[OFFSET_VAL]])
// SPIRV: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %[[HANDLE]], <3 x i32> %[[COORD]], i32 %[[LOD]], <2 x i32> %[[OFFSET_VAL]])
// CHECK: ret <4 x float> %[[RES]]


// For the rest of the types, we just check that the call to the member
// function has the correct return type.

Texture2DArray<float> t_float;

// CHECK: define hidden {{.*}} float @test_load_float(int vector[2])
// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::Texture2DArray<float>::Load(int vector[4])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call {{.*}} float @llvm.dx.resource.load.level.f32.tdx.Texture_f32_0_0_0_7t.v3i32.i32.v2i32(target("dx.Texture", float, 0, 0, 0, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call {{.*}} float @llvm.spv.resource.load.level.f32.tspirv.Image_f32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret float %[[RES]]
float test_load_float(int2 loc : LOC) {
  return t_float.Load(int4(loc, 0, 0));
}

// CHECK: define hidden {{.*}} float @test_load_offset_float(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} float @hlsl::Texture2DArray<float>::Load(int vector[4], int vector[2])(ptr {{.*}} @t_float, <4 x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret float %[[CALL]]
float test_load_offset_float(int2 loc : LOC) {
  return t_float.Load(int4(loc, 0, 0), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::Texture2DArray<float>::Load(int vector[4], int vector[2])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call {{.*}} float @llvm.dx.resource.load.level.f32.tdx.Texture_f32_0_0_0_7t.v3i32.i32.v2i32(target("dx.Texture", float, 0, 0, 0, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call {{.*}} float @llvm.spv.resource.load.level.f32.tspirv.Image_f32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret float %[[RES]]

Texture2DArray<float2> t_float2;

// CHECK: define hidden {{.*}} <2 x float> @test_load_float2(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <2 x float> @hlsl::Texture2DArray<float vector[2]>::Load(int vector[4])(ptr {{.*}} @t_float2, <4 x i32> noundef %{{.*}})
// CHECK: ret <2 x float> %[[CALL]]
float2 test_load_float2(int2 loc : LOC) {
  return t_float2.Load(int4(loc, 0, 0));
}

// CHECK: define linkonce_odr hidden {{.*}} <2 x float> @hlsl::Texture2DArray<float vector[2]>::Load(int vector[4])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call {{.*}} <2 x float> @llvm.dx.resource.load.level.v2f32.tdx.Texture_v2f32_0_0_0_7t.v3i32.i32.v2i32(target("dx.Texture", <2 x float>, 0, 0, 0, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call {{.*}} <2 x float> @llvm.spv.resource.load.level.v2f32.tspirv.Image_f32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret <2 x float> %[[RES]]

// CHECK: define hidden {{.*}} <2 x float> @test_load_offset_float2(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <2 x float> @hlsl::Texture2DArray<float vector[2]>::Load(int vector[4], int vector[2])(ptr {{.*}} @t_float2, <4 x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret <2 x float> %[[CALL]]
float2 test_load_offset_float2(int2 loc : LOC) {
  return t_float2.Load(int4(loc, 0, 0), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} <2 x float> @hlsl::Texture2DArray<float vector[2]>::Load(int vector[4], int vector[2])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call {{.*}} <2 x float> @llvm.dx.resource.load.level.v2f32.tdx.Texture_v2f32_0_0_0_7t.v3i32.i32.v2i32(target("dx.Texture", <2 x float>, 0, 0, 0, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call {{.*}} <2 x float> @llvm.spv.resource.load.level.v2f32.tspirv.Image_f32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret <2 x float> %[[RES]]

Texture2DArray<float3> t_float3;

// CHECK: define hidden {{.*}} <3 x float> @test_load_float3(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x float> @hlsl::Texture2DArray<float vector[3]>::Load(int vector[4])(ptr {{.*}} @t_float3, <4 x i32> noundef %{{.*}})
// CHECK: ret <3 x float> %[[CALL]]
float3 test_load_float3(int2 loc : LOC) {
  return t_float3.Load(int4(loc, 0, 0));
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x float> @hlsl::Texture2DArray<float vector[3]>::Load(int vector[4])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call {{.*}} <3 x float> @llvm.dx.resource.load.level.v3f32.tdx.Texture_v3f32_0_0_0_7t.v3i32.i32.v2i32(target("dx.Texture", <3 x float>, 0, 0, 0, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call {{.*}} <3 x float> @llvm.spv.resource.load.level.v3f32.tspirv.Image_f32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret <3 x float> %[[RES]]

// CHECK: define hidden {{.*}} <3 x float> @test_load_offset_float3(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x float> @hlsl::Texture2DArray<float vector[3]>::Load(int vector[4], int vector[2])(ptr {{.*}} @t_float3, <4 x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret <3 x float> %[[CALL]]
float3 test_load_offset_float3(int2 loc : LOC) {
  return t_float3.Load(int4(loc, 0, 0), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x float> @hlsl::Texture2DArray<float vector[3]>::Load(int vector[4], int vector[2])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call {{.*}} <3 x float> @llvm.dx.resource.load.level.v3f32.tdx.Texture_v3f32_0_0_0_7t.v3i32.i32.v2i32(target("dx.Texture", <3 x float>, 0, 0, 0, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call {{.*}} <3 x float> @llvm.spv.resource.load.level.v3f32.tspirv.Image_f32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret <3 x float> %[[RES]]

Texture2DArray<int> t_int;

// CHECK: define hidden {{.*}} i32 @test_load_int(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} i32 @hlsl::Texture2DArray<int>::Load(int vector[4])(ptr {{.*}} @t_int, <4 x i32> noundef %{{.*}})
// CHECK: ret i32 %[[CALL]]
int test_load_int(int2 loc : LOC) {
  return t_int.Load(int4(loc, 0, 0));
}

// CHECK: define linkonce_odr hidden {{.*}} i32 @hlsl::Texture2DArray<int>::Load(int vector[4])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call i32 @llvm.dx.resource.load.level.i32.tdx.Texture_i32_0_0_1_7t.v3i32.i32.v2i32(target("dx.Texture", i32, 0, 0, 1, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call i32 @llvm.spv.resource.load.level.i32.tspirv.SignedImage_i32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.SignedImage", i32, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret i32 %[[RES]]

// CHECK: define hidden {{.*}} i32 @test_load_offset_int(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} i32 @hlsl::Texture2DArray<int>::Load(int vector[4], int vector[2])(ptr {{.*}} @t_int, <4 x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret i32 %[[CALL]]
int test_load_offset_int(int2 loc : LOC) {
  return t_int.Load(int4(loc, 0, 0), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} i32 @hlsl::Texture2DArray<int>::Load(int vector[4], int vector[2])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call i32 @llvm.dx.resource.load.level.i32.tdx.Texture_i32_0_0_1_7t.v3i32.i32.v2i32(target("dx.Texture", i32, 0, 0, 1, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call i32 @llvm.spv.resource.load.level.i32.tspirv.SignedImage_i32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.SignedImage", i32, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret i32 %[[RES]]

Texture2DArray<int2> t_int2;

// CHECK: define hidden {{.*}} <2 x i32> @test_load_int2(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <2 x i32> @hlsl::Texture2DArray<int vector[2]>::Load(int vector[4])(ptr {{.*}} @t_int2, <4 x i32> noundef %{{.*}})
// CHECK: ret <2 x i32> %[[CALL]]
int2 test_load_int2(int2 loc : LOC) {
  return t_int2.Load(int4(loc, 0, 0));
}

// CHECK: define linkonce_odr hidden {{.*}} <2 x i32> @hlsl::Texture2DArray<int vector[2]>::Load(int vector[4])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call <2 x i32> @llvm.dx.resource.load.level.v2i32.tdx.Texture_v2i32_0_0_1_7t.v3i32.i32.v2i32(target("dx.Texture", <2 x i32>, 0, 0, 1, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call <2 x i32> @llvm.spv.resource.load.level.v2i32.tspirv.SignedImage_i32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.SignedImage", i32, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret <2 x i32> %[[RES]]

// CHECK: define hidden {{.*}} <2 x i32> @test_load_offset_int2(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <2 x i32> @hlsl::Texture2DArray<int vector[2]>::Load(int vector[4], int vector[2])(ptr {{.*}} @t_int2, <4 x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret <2 x i32> %[[CALL]]
int2 test_load_offset_int2(int2 loc : LOC) {
  return t_int2.Load(int4(loc, 0, 0), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} <2 x i32> @hlsl::Texture2DArray<int vector[2]>::Load(int vector[4], int vector[2])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call <2 x i32> @llvm.dx.resource.load.level.v2i32.tdx.Texture_v2i32_0_0_1_7t.v3i32.i32.v2i32(target("dx.Texture", <2 x i32>, 0, 0, 1, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call <2 x i32> @llvm.spv.resource.load.level.v2i32.tspirv.SignedImage_i32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.SignedImage", i32, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret <2 x i32> %[[RES]]

Texture2DArray<int3> t_int3;

// CHECK: define hidden {{.*}} <3 x i32> @test_load_int3(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x i32> @hlsl::Texture2DArray<int vector[3]>::Load(int vector[4])(ptr {{.*}} @t_int3, <4 x i32> noundef %{{.*}})
// CHECK: ret <3 x i32> %[[CALL]]
int3 test_load_int3(int2 loc : LOC) {
  return t_int3.Load(int4(loc, 0, 0));
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x i32> @hlsl::Texture2DArray<int vector[3]>::Load(int vector[4])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.load.level.v3i32.tdx.Texture_v3i32_0_0_1_7t.v3i32.i32.v2i32(target("dx.Texture", <3 x i32>, 0, 0, 1, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.load.level.v3i32.tspirv.SignedImage_i32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.SignedImage", i32, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret <3 x i32> %[[RES]]

// CHECK: define hidden {{.*}} <3 x i32> @test_load_offset_int3(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x i32> @hlsl::Texture2DArray<int vector[3]>::Load(int vector[4], int vector[2])(ptr {{.*}} @t_int3, <4 x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret <3 x i32> %[[CALL]]
int3 test_load_offset_int3(int2 loc : LOC) {
  return t_int3.Load(int4(loc, 0, 0), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x i32> @hlsl::Texture2DArray<int vector[3]>::Load(int vector[4], int vector[2])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.load.level.v3i32.tdx.Texture_v3i32_0_0_1_7t.v3i32.i32.v2i32(target("dx.Texture", <3 x i32>, 0, 0, 1, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.load.level.v3i32.tspirv.SignedImage_i32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.SignedImage", i32, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret <3 x i32> %[[RES]]

Texture2DArray<int4> t_int4;

// CHECK: define hidden {{.*}} <4 x i32> @test_load_int4(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x i32> @hlsl::Texture2DArray<int vector[4]>::Load(int vector[4])(ptr {{.*}} @t_int4, <4 x i32> noundef %{{.*}})
// CHECK: ret <4 x i32> %[[CALL]]
int4 test_load_int4(int2 loc : LOC) {
  return t_int4.Load(int4(loc, 0, 0));
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x i32> @hlsl::Texture2DArray<int vector[4]>::Load(int vector[4])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]])
// DXIL: %[[RES:.*]] = call <4 x i32> @llvm.dx.resource.load.level.v4i32.tdx.Texture_v4i32_0_0_1_7t.v3i32.i32.v2i32(target("dx.Texture", <4 x i32>, 0, 0, 1, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: %[[RES:.*]] = call <4 x i32> @llvm.spv.resource.load.level.v4i32.tspirv.SignedImage_i32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.SignedImage", i32, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// CHECK: ret <4 x i32> %[[RES]]

// CHECK: define hidden {{.*}} <4 x i32> @test_load_offset_int4(int vector[2])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x i32> @hlsl::Texture2DArray<int vector[4]>::Load(int vector[4], int vector[2])(ptr {{.*}} @t_int4, <4 x i32> noundef %{{.*}}, <2 x i32> noundef splat (i32 1))
// CHECK: ret <4 x i32> %[[CALL]]
int4 test_load_offset_int4(int2 loc : LOC) {
  return t_int4.Load(int4(loc, 0, 0), int2(1, 1));
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x i32> @hlsl::Texture2DArray<int vector[4]>::Load(int vector[4], int vector[2])(ptr {{.*}} %[[THIS:.*]], <4 x i32> {{.*}} %[[LOCATION:.*]], <2 x i32> {{.*}} %[[OFFSET:.*]])
// DXIL: %[[RES:.*]] = call <4 x i32> @llvm.dx.resource.load.level.v4i32.tdx.Texture_v4i32_0_0_1_7t.v3i32.i32.v2i32(target("dx.Texture", <4 x i32>, 0, 0, 1, 7) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: %[[RES:.*]] = call <4 x i32> @llvm.spv.resource.load.level.v4i32.tspirv.SignedImage_i32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.SignedImage", i32, 1, 2, 1, 0, 1, 0) %{{.*}}, <3 x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// CHECK: ret <4 x i32> %[[RES]]
