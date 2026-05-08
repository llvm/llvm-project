// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV

Texture2D<float4> Tex : register(t0);

// CHECK: define {{.*}} void @test_uint_dims()
// CHECK: call void @hlsl::Texture2D<float vector[4]>::GetDimensions(unsigned int&, unsigned int&)(ptr {{.*}} @Tex, ptr {{.*}}, ptr {{.*}})
void test_uint_dims() {
  uint w, h;
  Tex.GetDimensions(w, h);
}

// TODO: The test will have to be updated because the return type for the getdimensions intrinsic will no longer a overloaded.

// CHECK: define linkonce_odr hidden void @hlsl::Texture2D<float vector[4]>::GetDimensions(unsigned int&, unsigned int&)(ptr {{.*}} %[[THIS:.*]], ptr {{.*}} %[[WIDTH:.*]], ptr {{.*}} %[[HEIGHT:.*]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2D", ptr %[[THIS_VAL]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, 0, 0, 0, 2), ptr %[[HANDLE_GEP]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, 1, 2, 0, 0, 1, 0), ptr %[[HANDLE_GEP]]
// DXIL: %[[RES:.*]] = call <2 x i32> @llvm.dx.resource.getdimensions.xy.tdx.Texture_v4f32_0_0_0_2t(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %[[HANDLE]])
// SPIRV: %[[RES:.*]] = call <2 x i32> @llvm.spv.resource.getdimensions.xy.tspirv.Image_f32_1_2_0_0_1_0t(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE]])
// CHECK: %[[W_PTR:.*]] = load ptr, ptr %[[WIDTH]]
// CHECK: %[[W_VAL:.*]] = extractelement <2 x i32> %[[RES]], i64 0
// CHECK: store i32 %[[W_VAL]], ptr %[[W_PTR]]
// CHECK: %[[H_PTR:.*]] = load ptr, ptr %[[HEIGHT]]
// CHECK: %[[H_VAL:.*]] = extractelement <2 x i32> %[[RES]], i64 1
// CHECK: store i32 %[[H_VAL]], ptr %[[H_PTR]]

// CHECK: define {{.*}} void @test_uint_levels_dims{{.*}}(i32 noundef %[[MIP_LEVEL:.*]])
// CHECK: call void @hlsl::Texture2D<float vector[4]>::GetDimensions(unsigned int, unsigned int&, unsigned int&, unsigned int&)(ptr {{.*}} @Tex, i32 noundef %{{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
void test_uint_levels_dims(uint mipLevel) {
  uint w, h, l;
  Tex.GetDimensions(mipLevel, w, h, l);
}

// CHECK: define linkonce_odr hidden void @hlsl::Texture2D<float vector[4]>::GetDimensions(unsigned int, unsigned int&, unsigned int&, unsigned int&)(ptr {{.*}} %[[THIS:.*]], i32 {{.*}} %[[MIP:.*]], ptr {{.*}} %[[WIDTH:.*]], ptr {{.*}} %[[HEIGHT:.*]], ptr {{.*}} %[[LEVELS:.*]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2D", ptr %[[THIS_VAL]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, 0, 0, 0, 2), ptr %[[HANDLE_GEP]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, 1, 2, 0, 0, 1, 0), ptr %[[HANDLE_GEP]]
// CHECK: %[[MIP_VAL:.*]] = load i32, ptr %[[MIP]]
// DXIL: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.getdimensions.levels.xy.tdx.Texture_v4f32_0_0_0_2t(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %[[HANDLE]], i32 %[[MIP_VAL]])
// SPIRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.getdimensions.levels.xy.tspirv.Image_f32_1_2_0_0_1_0t(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE]], i32 %[[MIP_VAL]])
// CHECK: %[[W_PTR:.*]] = load ptr, ptr %[[WIDTH]]
// CHECK: %[[W_VAL:.*]] = extractelement <3 x i32> %[[RES]], i64 0
// CHECK: store i32 %[[W_VAL]], ptr %[[W_PTR]]
// CHECK: %[[H_PTR:.*]] = load ptr, ptr %[[HEIGHT]]
// CHECK: %[[H_VAL:.*]] = extractelement <3 x i32> %[[RES]], i64 1
// CHECK: store i32 %[[H_VAL]], ptr %[[H_PTR]]
// CHECK: %[[L_PTR:.*]] = load ptr, ptr %[[LEVELS]]
// CHECK: %[[L_VAL:.*]] = extractelement <3 x i32> %[[RES]], i64 2
// CHECK: store i32 %[[L_VAL]], ptr %[[L_PTR]]

// CHECK: define {{.*}} void @test_float_dims()
// CHECK: call void @hlsl::Texture2D<float vector[4]>::GetDimensions(float&, float&)(ptr {{.*}} @Tex, ptr {{.*}}, ptr {{.*}})
void test_float_dims() {
  float w, h;
  Tex.GetDimensions(w, h);
}

// CHECK: define linkonce_odr hidden void @hlsl::Texture2D<float vector[4]>::GetDimensions(float&, float&)(ptr {{.*}} %[[THIS:.*]], ptr {{.*}} %[[WIDTH:.*]], ptr {{.*}} %[[HEIGHT:.*]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2D", ptr %[[THIS_VAL]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, 0, 0, 0, 2), ptr %[[HANDLE_GEP]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, 1, 2, 0, 0, 1, 0), ptr %[[HANDLE_GEP]]
// DXIL: %[[RES:.*]] = call <2 x i32> @llvm.dx.resource.getdimensions.xy.tdx.Texture_v4f32_0_0_0_2t(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %[[HANDLE]])
// SPIRV: %[[RES:.*]] = call <2 x i32> @llvm.spv.resource.getdimensions.xy.tspirv.Image_f32_1_2_0_0_1_0t(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE]])
// CHECK: %[[W_PTR:.*]] = load ptr, ptr %[[WIDTH]]
// CHECK: %[[W_VAL:.*]] = extractelement <2 x i32> %[[RES]], i64 0
// CHECK: %[[W_F:.*]] = uitofp i32 %[[W_VAL]] to float
// CHECK: store float %[[W_F]], ptr %[[W_PTR]]
// CHECK: %[[H_PTR:.*]] = load ptr, ptr %[[HEIGHT]]
// CHECK: %[[H_VAL:.*]] = extractelement <2 x i32> %[[RES]], i64 1
// CHECK: %[[H_F:.*]] = uitofp i32 %[[H_VAL]] to float
// CHECK: store float %[[H_F]], ptr %[[H_PTR]]

// CHECK: define {{.*}} void @test_float_levels_dims{{.*}}(i32 noundef %[[MIP_LEVEL:.*]])
// CHECK: call void @hlsl::Texture2D<float vector[4]>::GetDimensions(unsigned int, float&, float&, float&)(ptr {{.*}} @Tex, i32 noundef %{{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
void test_float_levels_dims(uint mipLevel) {
  float w, h, l;
  Tex.GetDimensions(mipLevel, w, h, l);
}

// CHECK: define linkonce_odr hidden void @hlsl::Texture2D<float vector[4]>::GetDimensions(unsigned int, float&, float&, float&)(ptr {{.*}} %[[THIS:.*]], i32 {{.*}} %[[MIP:.*]], ptr {{.*}} %[[WIDTH:.*]], ptr {{.*}} %[[HEIGHT:.*]], ptr {{.*}} %[[LEVELS:.*]])
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::Texture2D", ptr %[[THIS_VAL]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, 0, 0, 0, 2), ptr %[[HANDLE_GEP]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, 1, 2, 0, 0, 1, 0), ptr %[[HANDLE_GEP]]
// CHECK: %[[MIP_VAL:.*]] = load i32, ptr %[[MIP]]
// DXIL: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.getdimensions.levels.xy.tdx.Texture_v4f32_0_0_0_2t(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %[[HANDLE]], i32 %[[MIP_VAL]])
// SPIRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.getdimensions.levels.xy.tspirv.Image_f32_1_2_0_0_1_0t(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE]], i32 %[[MIP_VAL]])
// CHECK: %[[W_PTR:.*]] = load ptr, ptr %[[WIDTH]]
// CHECK: %[[W_VAL:.*]] = extractelement <3 x i32> %[[RES]], i64 0
// CHECK: %[[W_F:.*]] = uitofp i32 %[[W_VAL]] to float
// CHECK: store float %[[W_F]], ptr %[[W_PTR]]
// CHECK: %[[H_PTR:.*]] = load ptr, ptr %[[HEIGHT]]
// CHECK: %[[H_VAL:.*]] = extractelement <3 x i32> %[[RES]], i64 1
// CHECK: %[[H_F:.*]] = uitofp i32 %[[H_VAL]] to float
// CHECK: store float %[[H_F]], ptr %[[H_PTR]]
// CHECK: %[[L_PTR:.*]] = load ptr, ptr %[[LEVELS]]
// CHECK: %[[L_VAL:.*]] = extractelement <3 x i32> %[[RES]], i64 2
// CHECK: %[[L_F:.*]] = uitofp i32 %[[L_VAL]] to float
// CHECK: store float %[[L_F]], ptr %[[L_PTR]]
