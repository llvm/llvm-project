// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -Wno-sign-conversion -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -Wno-sign-conversion -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV

Texture2D<float4> Tex : register(t0);
Texture2D<float> Tex2 : register(t1);
Texture2D<int3> Tex3 : register(t2);

[numthreads(1,1,1)]
void main(uint2 DTid : SV_DispatchThreadID) {
  float4 val = Tex[DTid];
  float val2 = Tex2[DTid];
  int3 val3 = Tex3[DTid];
}

// CHECK: define hidden {{.*}}void @main(unsigned int vector[2])(<2 x i32> noundef %[[DTID:.*]])
// CHECK: %[[DTID_ADDR:.*]] = alloca <2 x i32>
// CHECK: %[[VAL:.*]] = alloca <4 x float>
// CHECK: %[[VAL2:.*]] = alloca float
// CHECK: %[[VAL3:.*]] = alloca <3 x i32>
// CHECK: store <2 x i32> %[[DTID]], ptr %[[DTID_ADDR]]
// CHECK: %[[DTID_VAL:.*]] = load <2 x i32>, ptr %[[DTID_ADDR]]
// CHECK: %[[CALL1:.*]] = call noundef {{.*}}ptr{{.*}} @hlsl::Texture2D<float vector[4]>::operator[](unsigned int vector[2]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) @Tex, <2 x i32> noundef %[[DTID_VAL]])
// CHECK: %[[LOAD_VAL:.*]] = load <4 x float>, ptr{{.*}} %[[CALL1]]
// CHECK: store <4 x float> %[[LOAD_VAL]], ptr %[[VAL]]
// CHECK: %[[DTID_VAL2:.*]] = load <2 x i32>, ptr %[[DTID_ADDR]]
// CHECK: %[[CALL2:.*]] = call noundef {{.*}}ptr{{.*}} @hlsl::Texture2D<float>::operator[](unsigned int vector[2]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) @Tex2, <2 x i32> noundef %[[DTID_VAL2]])
// CHECK: %[[LOAD_VAL2:.*]] = load float, ptr{{.*}} %[[CALL2]]
// CHECK: store float %[[LOAD_VAL2]], ptr %[[VAL2]]
// CHECK: %[[DTID_VAL3:.*]] = load <2 x i32>, ptr %[[DTID_ADDR]]
// CHECK: %[[CALL3:.*]] = call noundef {{.*}}ptr{{.*}} @hlsl::Texture2D<int vector[3]>::operator[](unsigned int vector[2]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) @Tex3, <2 x i32> noundef %[[DTID_VAL3]])
// CHECK: %[[LOAD_VAL3:.*]] = load <3 x i32>, ptr{{.*}} %[[CALL3]]
// CHECK: store <3 x i32> %[[LOAD_VAL3]], ptr %[[VAL3]]

// CHECK: define linkonce_odr hidden noundef {{.*}}ptr{{.*}} @hlsl::Texture2D<float vector[4]>::operator[](unsigned int vector[2]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %[[THIS:.*]], <2 x i32> noundef %[[INDEX:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[INDEX_ADDR:.*]] = alloca <2 x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <2 x i32> %[[INDEX]], ptr %[[INDEX_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr {{.*}} %"class.hlsl::Texture2D", ptr %[[THIS1]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, 0, 0, 0, 2), ptr %[[HANDLE_PTR]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, 1, 2, 0, 0, 1, 0), ptr %[[HANDLE_PTR]]
// CHECK: %[[INDEX_VAL:.*]] = load <2 x i32>, ptr %[[INDEX_ADDR]]
// DXIL: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.Texture_v4f32_0_0_0_2t.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %[[HANDLE]], <2 x i32> %[[INDEX_VAL]])
// SPIRV: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_1_2_0_0_1_0t.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE]], <2 x i32> %[[INDEX_VAL]])
// CHECK: ret ptr {{.*}}%[[PTR]]

// CHECK: define linkonce_odr hidden noundef {{.*}}ptr{{.*}} @hlsl::Texture2D<float>::operator[](unsigned int vector[2]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %[[THIS:.*]], <2 x i32> noundef %[[INDEX:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[INDEX_ADDR:.*]] = alloca <2 x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <2 x i32> %[[INDEX]], ptr %[[INDEX_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr {{.*}} %"class.hlsl::Texture2D.0", ptr %[[THIS1]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", float, 0, 0, 0, 2), ptr %[[HANDLE_PTR]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, 1, 2, 0, 0, 1, 0), ptr %[[HANDLE_PTR]]
// CHECK: %[[INDEX_VAL:.*]] = load <2 x i32>, ptr %[[INDEX_ADDR]]
// DXIL: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.Texture_f32_0_0_0_2t.v2i32(target("dx.Texture", float, 0, 0, 0, 2) %[[HANDLE]], <2 x i32> %[[INDEX_VAL]])
// SPIRV: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_1_2_0_0_1_0t.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %[[HANDLE]], <2 x i32> %[[INDEX_VAL]])
// CHECK: ret ptr {{.*}}%[[PTR]]

// CHECK: define linkonce_odr hidden noundef {{.*}}ptr{{.*}} @hlsl::Texture2D<int vector[3]>::operator[](unsigned int vector[2]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %[[THIS:.*]], <2 x i32> noundef %[[INDEX:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[INDEX_ADDR:.*]] = alloca <2 x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <2 x i32> %[[INDEX]], ptr %[[INDEX_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr {{.*}} %"class.hlsl::Texture2D.1", ptr %[[THIS1]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <3 x i32>, 0, 0, 1, 2), ptr %[[HANDLE_PTR]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.SignedImage", i32, 1, 2, 0, 0, 1, 0), ptr %[[HANDLE_PTR]]
// CHECK: %[[INDEX_VAL:.*]] = load <2 x i32>, ptr %[[INDEX_ADDR]]
// DXIL: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.Texture_v3i32_0_0_1_2t.v2i32(target("dx.Texture", <3 x i32>, 0, 0, 1, 2) %[[HANDLE]], <2 x i32> %[[INDEX_VAL]])
// SPIRV: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_1_2_0_0_1_0t.v2i32(target("spirv.SignedImage", i32, 1, 2, 0, 0, 1, 0) %[[HANDLE]], <2 x i32> %[[INDEX_VAL]])
// CHECK: ret ptr {{.*}}%[[PTR]]
