// Texture2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2D -o \
// RUN:   - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=Texture2D \
// RUN:   -DDXIL_TY=2 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2D -o \
// RUN:   - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=Texture2D \
// RUN:   -DARRAYED=0 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1

// Texture2DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=Texture2DArray -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=Texture2DArray \
// RUN:   -DDXIL_TY=7 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=Texture2DArray -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=Texture2DArray \
// RUN:   -DARRAYED=1 -DSAMPLED=1 -DIMG_FMT=0 -DSPV_DIM=1

// RWTexture2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DTEXTURE=RWTexture2D \
// RUN:   -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=RWTexture2D \
// RUN:   -DDXIL_TY=2 -DRW=1
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DTEXTURE=RWTexture2D \
// RUN:   -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=RWTexture2D \
// RUN:   -DARRAYED=0 -DSAMPLED=2 -DIMG_FMT=1 -DSPV_DIM=1

// RWTexture2DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=RWTexture2DArray -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,DXIL \
// RUN:   -DTEXTURE=RWTexture2DArray -DDXIL_TY=7 -DRW=1
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=RWTexture2DArray -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SPIRV \
// RUN:   -DTEXTURE=RWTexture2DArray -DARRAYED=1 -DSAMPLED=2 -DIMG_FMT=1 \
// RUN:   -DSPV_DIM=1

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   DXIL_TY            dx.Texture resource-kind operand
//   RW                 dx.Texture UAV operand
//   ARRAYED            spirv.Image Arrayed operand
//   SAMPLED            spirv.Image Sampled operand
//   IMG_FMT            spirv.Image Image Format operand
//   SPV_DIM            spirv.Image Dim operand

TEXTURE<float4> Tex;

// CHECK: define {{.*}} void @test_uint_dims{{(\(\))?}}()
// CHECK: call {{(spir_func )?}}void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(unsigned int&, unsigned int&)(ptr {{.*}} @Tex, ptr {{.*}}, ptr {{.*}})
void test_uint_dims() {
  uint w, h;
  Tex.GetDimensions(w, h);
}

// CHECK: define linkonce_odr hidden {{(spir_func )?}}void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(unsigned int&, unsigned int&)(ptr {{.*}} %[[THIS:.*]], ptr {{.*}} %[[WIDTH:.*]], ptr {{.*}} %[[HEIGHT:.*]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), ptr %[[HANDLE_GEP]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]), ptr %[[HANDLE_GEP]]
// DXIL: %[[RES:.*]] = call <2 x i32> @llvm.dx.resource.getdimensions.xy.tdx.Texture_v4f32_{{.*}}("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]])
// SPIRV: %[[RES:.*]] = call <2 x i32> @llvm.spv.resource.getdimensions.xy.tspirv.Image_f32_{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]])
// CHECK: %[[W_PTR:.*]] = load ptr, ptr %[[WIDTH]]
// CHECK: %[[W_VAL:.*]] = extractelement <2 x i32> %[[RES]], i64 0
// CHECK: store i32 %[[W_VAL]], ptr %[[W_PTR]]
// CHECK: %[[H_PTR:.*]] = load ptr, ptr %[[HEIGHT]]
// CHECK: %[[H_VAL:.*]] = extractelement <2 x i32> %[[RES]], i64 1
// CHECK: store i32 %[[H_VAL]], ptr %[[H_PTR]]

// CHECK: define {{.*}} void @test_uint_levels_dims{{.*}}(i32 noundef %{{.*}})
// CHECK: call {{(spir_func )?}}void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(unsigned int, unsigned int&, unsigned int&, unsigned int&)(ptr {{.*}} @Tex, i32 noundef %{{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
void test_uint_levels_dims(uint mipLevel) {
  uint w, h, l;
  Tex.GetDimensions(mipLevel, w, h, l);
}

// CHECK: define linkonce_odr hidden {{(spir_func )?}}void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(unsigned int, unsigned int&, unsigned int&, unsigned int&)(ptr {{.*}} %[[THIS:.*]], i32 {{.*}} %[[MIP:.*]], ptr {{.*}} %[[WIDTH:.*]], ptr {{.*}} %[[HEIGHT:.*]], ptr {{.*}} %[[LEVELS:.*]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), ptr %[[HANDLE_GEP]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]), ptr %[[HANDLE_GEP]]
// CHECK: %[[MIP_VAL:.*]] = load i32, ptr %[[MIP]]
// DXIL: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.getdimensions.levels.xy.tdx.Texture_v4f32_{{.*}}("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], i32 %[[MIP_VAL]])
// SPIRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.getdimensions.levels.xy.tspirv.Image_f32_{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], i32 %[[MIP_VAL]])
// CHECK: %[[W_PTR:.*]] = load ptr, ptr %[[WIDTH]]
// CHECK: %[[W_VAL:.*]] = extractelement <3 x i32> %[[RES]], i64 0
// CHECK: store i32 %[[W_VAL]], ptr %[[W_PTR]]
// CHECK: %[[H_PTR:.*]] = load ptr, ptr %[[HEIGHT]]
// CHECK: %[[H_VAL:.*]] = extractelement <3 x i32> %[[RES]], i64 1
// CHECK: store i32 %[[H_VAL]], ptr %[[H_PTR]]
// CHECK: %[[L_PTR:.*]] = load ptr, ptr %[[LEVELS]]
// CHECK: %[[L_VAL:.*]] = extractelement <3 x i32> %[[RES]], i64 2
// CHECK: store i32 %[[L_VAL]], ptr %[[L_PTR]]

// CHECK: define {{.*}} void @test_float_dims{{(\(\))?}}()
// CHECK: call {{(spir_func )?}}void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(float&, float&)(ptr {{.*}} @Tex, ptr {{.*}}, ptr {{.*}})
void test_float_dims() {
  float w, h;
  Tex.GetDimensions(w, h);
}

// CHECK: define linkonce_odr hidden {{(spir_func )?}}void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(float&, float&)(ptr {{.*}} %[[THIS:.*]], ptr {{.*}} %[[WIDTH:.*]], ptr {{.*}} %[[HEIGHT:.*]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), ptr %[[HANDLE_GEP]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]), ptr %[[HANDLE_GEP]]
// DXIL: %[[RES:.*]] = call <2 x i32> @llvm.dx.resource.getdimensions.xy.tdx.Texture_v4f32_{{.*}}("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]])
// SPIRV: %[[RES:.*]] = call <2 x i32> @llvm.spv.resource.getdimensions.xy.tspirv.Image_f32_{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]])
// CHECK: %[[W_PTR:.*]] = load ptr, ptr %[[WIDTH]]
// CHECK: %[[W_VAL:.*]] = extractelement <2 x i32> %[[RES]], i64 0
// CHECK: %[[W_F:.*]] = uitofp reassoc nnan ninf nsz arcp afn i32 %[[W_VAL]] to float
// CHECK: store float %[[W_F]], ptr %[[W_PTR]]
// CHECK: %[[H_PTR:.*]] = load ptr, ptr %[[HEIGHT]]
// CHECK: %[[H_VAL:.*]] = extractelement <2 x i32> %[[RES]], i64 1
// CHECK: %[[H_F:.*]] = uitofp reassoc nnan ninf nsz arcp afn i32 %[[H_VAL]] to float
// CHECK: store float %[[H_F]], ptr %[[H_PTR]]

// CHECK: define {{.*}} void @test_float_levels_dims{{.*}}(i32 noundef %{{.*}})
// CHECK: call {{(spir_func )?}}void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(unsigned int, float&, float&, float&)(ptr {{.*}} @Tex, i32 noundef %{{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
void test_float_levels_dims(uint mipLevel) {
  float w, h, l;
  Tex.GetDimensions(mipLevel, w, h, l);
}

// CHECK: define linkonce_odr hidden {{(spir_func )?}}void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(unsigned int, float&, float&, float&)(ptr {{.*}} %[[THIS:.*]], i32 {{.*}} %[[MIP:.*]], ptr {{.*}} %[[WIDTH:.*]], ptr {{.*}} %[[HEIGHT:.*]], ptr {{.*}} %[[LEVELS:.*]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), ptr %[[HANDLE_GEP]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]), ptr %[[HANDLE_GEP]]
// CHECK: %[[MIP_VAL:.*]] = load i32, ptr %[[MIP]]
// DXIL: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.getdimensions.levels.xy.tdx.Texture_v4f32_{{.*}}("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], i32 %[[MIP_VAL]])
// SPIRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.getdimensions.levels.xy.tspirv.Image_f32_{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], i32 %[[MIP_VAL]])
// CHECK: %[[W_PTR:.*]] = load ptr, ptr %[[WIDTH]]
// CHECK: %[[W_VAL:.*]] = extractelement <3 x i32> %[[RES]], i64 0
// CHECK: %[[W_F:.*]] = uitofp reassoc nnan ninf nsz arcp afn i32 %[[W_VAL]] to float
// CHECK: store float %[[W_F]], ptr %[[W_PTR]]
// CHECK: %[[H_PTR:.*]] = load ptr, ptr %[[HEIGHT]]
// CHECK: %[[H_VAL:.*]] = extractelement <3 x i32> %[[RES]], i64 1
// CHECK: %[[H_F:.*]] = uitofp reassoc nnan ninf nsz arcp afn i32 %[[H_VAL]] to float
// CHECK: store float %[[H_F]], ptr %[[H_PTR]]
// CHECK: %[[L_PTR:.*]] = load ptr, ptr %[[LEVELS]]
// CHECK: %[[L_VAL:.*]] = extractelement <3 x i32> %[[RES]], i64 2
// CHECK: %[[L_F:.*]] = uitofp reassoc nnan ninf nsz arcp afn i32 %[[L_VAL]] to float
// CHECK: store float %[[L_F]], ptr %[[L_PTR]]
