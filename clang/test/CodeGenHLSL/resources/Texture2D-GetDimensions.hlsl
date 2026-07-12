// Texture2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2D -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=Texture2D -DDXILTY=2 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2D -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=Texture2D -DARRAYED=0 -DSAMPLED=1 -DFORMAT=0

// Texture2DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DArray -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=Texture2DArray -DDXILTY=7 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DArray -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=Texture2DArray -DARRAYED=1 -DSAMPLED=1 -DFORMAT=0

// RWTexture2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=RWTexture2D -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=RWTexture2D -DDXILTY=2 -DRW=1
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=RWTexture2D -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=RWTexture2D -DARRAYED=0 -DSAMPLED=2 -DFORMAT=1

// When RWTexture2DArray is implemented, add DXIL/SPIRV runs with DXILTY=7, RW=1, ARRAYED=1, SAMPLED=2, FORMAT=1.

TEXTURE<float4> Tex;

// CHECK: define {{.*}} void @test_uint_dims{{(\(\))?}}()
// CHECK: call void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(unsigned int&, unsigned int&)(ptr {{.*}} @Tex, ptr {{.*}}, ptr {{.*}})
void test_uint_dims() {
  uint w, h;
  Tex.GetDimensions(w, h);
}

// CHECK: define linkonce_odr hidden void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(unsigned int&, unsigned int&)(ptr {{.*}} %[[THIS:.*]], ptr {{.*}} %[[WIDTH:.*]], ptr {{.*}} %[[HEIGHT:.*]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXILTY]]), ptr %[[HANDLE_GEP]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT]]), ptr %[[HANDLE_GEP]]
// DXIL: %[[RES:.*]] = call <2 x i32> @llvm.dx.resource.getdimensions.xy.tdx.Texture_v4f32_{{.*}}("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXILTY]]) %[[HANDLE]])
// SPIRV: %[[RES:.*]] = call <2 x i32> @llvm.spv.resource.getdimensions.xy.tspirv.Image_f32_{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT]]) %[[HANDLE]])
// CHECK: %[[W_PTR:.*]] = load ptr, ptr %[[WIDTH]]
// CHECK: %[[W_VAL:.*]] = extractelement <2 x i32> %[[RES]], i64 0
// CHECK: store i32 %[[W_VAL]], ptr %[[W_PTR]]
// CHECK: %[[H_PTR:.*]] = load ptr, ptr %[[HEIGHT]]
// CHECK: %[[H_VAL:.*]] = extractelement <2 x i32> %[[RES]], i64 1
// CHECK: store i32 %[[H_VAL]], ptr %[[H_PTR]]

// CHECK: define {{.*}} void @test_uint_levels_dims{{.*}}(i32 noundef %{{.*}})
// CHECK: call void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(unsigned int, unsigned int&, unsigned int&, unsigned int&)(ptr {{.*}} @Tex, i32 noundef %{{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
void test_uint_levels_dims(uint mipLevel) {
  uint w, h, l;
  Tex.GetDimensions(mipLevel, w, h, l);
}

// CHECK: define linkonce_odr hidden void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(unsigned int, unsigned int&, unsigned int&, unsigned int&)(ptr {{.*}} %[[THIS:.*]], i32 {{.*}} %[[MIP:.*]], ptr {{.*}} %[[WIDTH:.*]], ptr {{.*}} %[[HEIGHT:.*]], ptr {{.*}} %[[LEVELS:.*]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXILTY]]), ptr %[[HANDLE_GEP]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT]]), ptr %[[HANDLE_GEP]]
// CHECK: %[[MIP_VAL:.*]] = load i32, ptr %[[MIP]]
// DXIL: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.getdimensions.levels.xy.tdx.Texture_v4f32_{{.*}}("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXILTY]]) %[[HANDLE]], i32 %[[MIP_VAL]])
// SPIRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.getdimensions.levels.xy.tspirv.Image_f32_{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT]]) %[[HANDLE]], i32 %[[MIP_VAL]])
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
// CHECK: call void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(float&, float&)(ptr {{.*}} @Tex, ptr {{.*}}, ptr {{.*}})
void test_float_dims() {
  float w, h;
  Tex.GetDimensions(w, h);
}

// CHECK: define linkonce_odr hidden void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(float&, float&)(ptr {{.*}} %[[THIS:.*]], ptr {{.*}} %[[WIDTH:.*]], ptr {{.*}} %[[HEIGHT:.*]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXILTY]]), ptr %[[HANDLE_GEP]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT]]), ptr %[[HANDLE_GEP]]
// DXIL: %[[RES:.*]] = call <2 x i32> @llvm.dx.resource.getdimensions.xy.tdx.Texture_v4f32_{{.*}}("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXILTY]]) %[[HANDLE]])
// SPIRV: %[[RES:.*]] = call <2 x i32> @llvm.spv.resource.getdimensions.xy.tspirv.Image_f32_{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT]]) %[[HANDLE]])
// CHECK: %[[W_PTR:.*]] = load ptr, ptr %[[WIDTH]]
// CHECK: %[[W_VAL:.*]] = extractelement <2 x i32> %[[RES]], i64 0
// CHECK: %[[W_F:.*]] = uitofp reassoc nnan ninf nsz arcp afn i32 %[[W_VAL]] to float
// CHECK: store float %[[W_F]], ptr %[[W_PTR]]
// CHECK: %[[H_PTR:.*]] = load ptr, ptr %[[HEIGHT]]
// CHECK: %[[H_VAL:.*]] = extractelement <2 x i32> %[[RES]], i64 1
// CHECK: %[[H_F:.*]] = uitofp reassoc nnan ninf nsz arcp afn i32 %[[H_VAL]] to float
// CHECK: store float %[[H_F]], ptr %[[H_PTR]]

// CHECK: define {{.*}} void @test_float_levels_dims{{.*}}(i32 noundef %{{.*}})
// CHECK: call void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(unsigned int, float&, float&, float&)(ptr {{.*}} @Tex, i32 noundef %{{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
void test_float_levels_dims(uint mipLevel) {
  float w, h, l;
  Tex.GetDimensions(mipLevel, w, h, l);
}

// CHECK: define linkonce_odr hidden void @hlsl::[[TEXTURE]]<float vector[4]>::GetDimensions(unsigned int, float&, float&, float&)(ptr {{.*}} %[[THIS:.*]], i32 {{.*}} %[[MIP:.*]], ptr {{.*}} %[[WIDTH:.*]], ptr {{.*}} %[[HEIGHT:.*]], ptr {{.*}} %[[LEVELS:.*]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXILTY]]), ptr %[[HANDLE_GEP]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT]]), ptr %[[HANDLE_GEP]]
// CHECK: %[[MIP_VAL:.*]] = load i32, ptr %[[MIP]]
// DXIL: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.getdimensions.levels.xy.tdx.Texture_v4f32_{{.*}}("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXILTY]]) %[[HANDLE]], i32 %[[MIP_VAL]])
// SPIRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.getdimensions.levels.xy.tspirv.Image_f32_{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT]]) %[[HANDLE]], i32 %[[MIP_VAL]])
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
