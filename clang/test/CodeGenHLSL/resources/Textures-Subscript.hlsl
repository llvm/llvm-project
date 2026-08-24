// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -Wno-sign-conversion -DTEXTURE=Texture2D -DCOORD_TYPE=uint2 -o - %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 --check-prefixes=CHECK,DXIL -DROV_OR_COUNT=0 -DDXIL_HANDLE=dx.Texture -DDXIL_TY=2 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -Wno-sign-conversion -DTEXTURE=Texture2D -DCOORD_TYPE=uint2 -o - %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 --check-prefixes=CHECK,SPIRV -DARRAYED=0 -DMS=0 -DSAMPLED=1 -DIMG_FMT=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -Wno-sign-conversion -DTEXTURE=Texture2DArray -DCOORD_TYPE=uint3 -o - %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 --check-prefixes=CHECK,DXIL -DROV_OR_COUNT=0 -DDXIL_HANDLE=dx.Texture -DDXIL_TY=7 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -Wno-sign-conversion -DTEXTURE=Texture2DArray -DCOORD_TYPE=uint3 -o - %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 --check-prefixes=CHECK,SPIRV -DARRAYED=1 -DMS=0 -DSAMPLED=1 -DIMG_FMT=0

// Texture2DMS reuses the same operator[] codegen; only the resource handle type
// differs (dx.MSTexture / a multisampled spirv.Image). It reads sample 0.
//
// ROV_OR_COUNT is the overloaded second int operand of the DXIL handle type:
// IsROV for dx.Texture, the sample count for dx.MSTexture. It is 0 for every
// texture below (none are ROVs, and Texture2DMS<T> defaults to a runtime
// sample count), but a ROV or an explicit Texture2DMS<T, N> would need its own
// value here.
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -Wno-sign-conversion -DTEXTURE=Texture2DMS -DCOORD_TYPE=uint2 -o - %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2DMS -DCOORD_DIM=2 --check-prefixes=CHECK,DXIL -DROV_OR_COUNT=0 -DDXIL_HANDLE=dx.MSTexture -DDXIL_TY=3 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -Wno-sign-conversion -DTEXTURE=Texture2DMS -DCOORD_TYPE=uint2 -o - %s | llvm-cxxfilt | FileCheck %s -DTEXTURE=Texture2DMS -DCOORD_DIM=2 --check-prefixes=CHECK,SPIRV -DARRAYED=0 -DMS=1 -DSAMPLED=1 -DIMG_FMT=0

TEXTURE<float4> Tex : register(t0);
TEXTURE<float> Tex2 : register(t1);
TEXTURE<int3> Tex3 : register(t2);

[numthreads(1,1,1)]
void main(COORD_TYPE DTid : SV_DispatchThreadID) {
  float4 val = Tex[DTid];
  float val2 = Tex2[DTid];
  int3 val3 = Tex3[DTid];
}

// CHECK: define hidden {{.*}}void @main(unsigned int vector[[[COORD_DIM]]])(<[[COORD_DIM]] x i32> noundef %[[DTID:.*]])
// CHECK: %[[DTID_ADDR:.*]] = alloca <[[COORD_DIM]] x i32>
// CHECK: %[[VAL:.*]] = alloca <4 x float>
// CHECK: %[[VAL2:.*]] = alloca float
// CHECK: %[[VAL3:.*]] = alloca <3 x i32>
// CHECK: store <[[COORD_DIM]] x i32> %[[DTID]], ptr %[[DTID_ADDR]]
// CHECK: %[[DTID_VAL:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[DTID_ADDR]]
// CHECK: %[[CALL1:.*]] = call {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<float vector[4]{{(, [0-9]+)?}}>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) @Tex, <[[COORD_DIM]] x i32> noundef %[[DTID_VAL]])
// CHECK: %[[LOAD_VAL:.*]] = load <4 x float>, ptr{{.*}} %[[CALL1]]
// CHECK: store <4 x float> %[[LOAD_VAL]], ptr %[[VAL]]
// CHECK: %[[DTID_VAL2:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[DTID_ADDR]]
// CHECK: %[[CALL2:.*]] = call {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<float{{(, [0-9]+)?}}>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) @Tex2, <[[COORD_DIM]] x i32> noundef %[[DTID_VAL2]])
// CHECK: %[[LOAD_VAL2:.*]] = load float, ptr{{.*}} %[[CALL2]]
// CHECK: store float %[[LOAD_VAL2]], ptr %[[VAL2]]
// CHECK: %[[DTID_VAL3:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[DTID_ADDR]]
// CHECK: %[[CALL3:.*]] = call {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<int vector[3]{{(, [0-9]+)?}}>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) @Tex3, <[[COORD_DIM]] x i32> noundef %[[DTID_VAL3]])
// CHECK: %[[LOAD_VAL3:.*]] = load <3 x i32>, ptr{{.*}} %[[CALL3]]
// CHECK: store <3 x i32> %[[LOAD_VAL3]], ptr %[[VAL3]]

// CHECK: define linkonce_odr hidden {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<float vector[4]{{(, [0-9]+)?}}>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %[[THIS:.*]], <[[COORD_DIM]] x i32> noundef %[[INDEX:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[INDEX_ADDR:.*]] = alloca <[[COORD_DIM]] x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x i32> %[[INDEX]], ptr %[[INDEX_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr {{.*}} %"class.hlsl::[[TEXTURE]]", ptr %[[THIS1]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("[[DXIL_HANDLE]]", <4 x float>, [[RW]], [[ROV_OR_COUNT]], 0, [[DXIL_TY]]), ptr %[[HANDLE_PTR]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, 1, 2, [[ARRAYED]], [[MS]], [[SAMPLED]], [[IMG_FMT]]), ptr %[[HANDLE_PTR]]
// CHECK: %[[INDEX_VAL:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[INDEX_ADDR]]
// DXIL: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.{{.*}}(target("[[DXIL_HANDLE]]", <4 x float>, [[RW]], [[ROV_OR_COUNT]], 0, [[DXIL_TY]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// SPIRV: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], [[MS]], [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// CHECK: ret ptr {{.*}}%[[PTR]]

// CHECK: define linkonce_odr hidden {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<float{{(, [0-9]+)?}}>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %[[THIS:.*]], <[[COORD_DIM]] x i32> noundef %[[INDEX:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[INDEX_ADDR:.*]] = alloca <[[COORD_DIM]] x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x i32> %[[INDEX]], ptr %[[INDEX_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr {{.*}} %"class.hlsl::[[TEXTURE]].0", ptr %[[THIS1]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("[[DXIL_HANDLE]]", float, [[RW]], [[ROV_OR_COUNT]], 0, [[DXIL_TY]]), ptr %[[HANDLE_PTR]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, 1, 2, [[ARRAYED]], [[MS]], [[SAMPLED]], [[IMG_FMT]]), ptr %[[HANDLE_PTR]]
// CHECK: %[[INDEX_VAL:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[INDEX_ADDR]]
// DXIL: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.{{.*}}(target("[[DXIL_HANDLE]]", float, [[RW]], [[ROV_OR_COUNT]], 0, [[DXIL_TY]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// SPIRV: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], [[MS]], [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// CHECK: ret ptr {{.*}}%[[PTR]]

// CHECK: define linkonce_odr hidden {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<int vector[3]{{(, [0-9]+)?}}>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %[[THIS:.*]], <[[COORD_DIM]] x i32> noundef %[[INDEX:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[INDEX_ADDR:.*]] = alloca <[[COORD_DIM]] x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x i32> %[[INDEX]], ptr %[[INDEX_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr {{.*}} %"class.hlsl::[[TEXTURE]].1", ptr %[[THIS1]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("[[DXIL_HANDLE]]", <3 x i32>, [[RW]], [[ROV_OR_COUNT]], 1, [[DXIL_TY]]), ptr %[[HANDLE_PTR]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], [[MS]], [[SAMPLED]], [[IMG_FMT]]), ptr %[[HANDLE_PTR]]
// CHECK: %[[INDEX_VAL:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[INDEX_ADDR]]
// DXIL: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.{{.*}}(target("[[DXIL_HANDLE]]", <3 x i32>, [[RW]], [[ROV_OR_COUNT]], 1, [[DXIL_TY]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// SPIRV: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{{.*}}(target("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], [[MS]], [[SAMPLED]], [[IMG_FMT]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// CHECK: ret ptr {{.*}}%[[PTR]]
