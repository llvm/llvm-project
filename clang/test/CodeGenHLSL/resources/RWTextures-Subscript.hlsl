// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DTEXTURE=RWTexture2D -DCOORD_TYPE=uint2 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=RWTexture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL -DDXIL_TY=2
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DTEXTURE=RWTexture2D -DCOORD_TYPE=uint2 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=RWTexture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,SPIRV -DARRAYED=0 -DSPV_DIM=1
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DTEXTURE=RWTexture2DArray -DCOORD_TYPE=uint3 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=RWTexture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL -DDXIL_TY=7
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DTEXTURE=RWTexture2DArray -DCOORD_TYPE=uint3 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=RWTexture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,SPIRV -DARRAYED=1 -DSPV_DIM=1

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//   COORD_DIM          sample location components (DIM plus the array slice)
//   DXIL_TY            dx.Texture resource-kind operand
//   ARRAYED            spirv.Image Arrayed operand
//   SPV_DIM            spirv.Image Dim operand
//
// `Tex[coord] = value` on a writable texture lowers to a `store` through the
// pointer from `resource.getpointer`, which the backends turn into
// `dx.op.textureStore` / `OpImageWrite`.

TEXTURE<float4> Tex : register(u0);
TEXTURE<float> Tex2 : register(u1);
TEXTURE<int3> Tex3 : register(u2);

[numthreads(1,1,1)]
void main(COORD_TYPE DTid : SV_DispatchThreadID) {
  Tex[DTid] = float4(1, 2, 3, 4);
  Tex[DTid].y = 5.0;
  Tex2[DTid] = 6.0;
  Tex3[DTid] = int3(7, 8, 9);
}

// CHECK: define hidden {{.*}}void @main(unsigned int vector[[[COORD_DIM]]])(<[[COORD_DIM]] x i32> noundef %[[DTID:.*]])
// CHECK: %[[DTID_ADDR:.*]] = alloca <[[COORD_DIM]] x i32>
// CHECK: store <[[COORD_DIM]] x i32> %[[DTID]], ptr %[[DTID_ADDR]]

// Store a whole texel.
// CHECK: %[[DTID_VAL:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[DTID_ADDR]]
// CHECK: %[[CALL1:.*]] = call {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<float vector[4]>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr {{.*}} @Tex, <[[COORD_DIM]] x i32> noundef %[[DTID_VAL]])
// CHECK: store <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>, ptr{{.*}} %[[CALL1]]

// Store a single component: a GEP off the texel pointer, which the DXIL
// backend later expands into a read-modify-write of the whole texel.
// CHECK: %[[DTID_VAL2:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[DTID_ADDR]]
// CHECK: %[[CALL2:.*]] = call {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<float vector[4]>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr {{.*}} @Tex, <[[COORD_DIM]] x i32> noundef %[[DTID_VAL2]])
// CHECK: %[[ELEM:.*]] = getelementptr <4 x float>, ptr{{.*}} %[[CALL2]], i32 0, i32 1
// CHECK: store float 5.000000e+00, ptr{{.*}} %[[ELEM]]

// Store to a scalar texture.
// CHECK: %[[DTID_VAL3:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[DTID_ADDR]]
// CHECK: %[[CALL3:.*]] = call {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<float>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr {{.*}} @Tex2, <[[COORD_DIM]] x i32> noundef %[[DTID_VAL3]])
// CHECK: store float 6.000000e+00, ptr{{.*}} %[[CALL3]]

// Store to an integer texture.
// CHECK: %[[DTID_VAL4:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[DTID_ADDR]]
// CHECK: %[[CALL4:.*]] = call {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<int vector[3]>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr {{.*}} @Tex3, <[[COORD_DIM]] x i32> noundef %[[DTID_VAL4]])
// CHECK: store <3 x i32> <i32 7, i32 8, i32 9>, ptr{{.*}} %[[CALL4]]

// Check the operator[] body
// CHECK: define linkonce_odr hidden {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<float vector[4]>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr {{.*}} %[[THIS:.*]], <[[COORD_DIM]] x i32> noundef %[[INDEX:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[INDEX_ADDR:.*]] = alloca <[[COORD_DIM]] x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x i32> %[[INDEX]], ptr %[[INDEX_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr {{.*}} %"class.hlsl::[[TEXTURE]]", ptr %[[THIS1]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, 1, 0, 0, [[DXIL_TY]]), ptr %[[HANDLE_PTR]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, 2, 1), ptr %[[HANDLE_PTR]]
// CHECK: %[[INDEX_VAL:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[INDEX_ADDR]]
// DXIL: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.{{.*}}(target("dx.Texture", <4 x float>, 1, 0, 0, [[DXIL_TY]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// SPIRV: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, 2, 1) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// CHECK: ret ptr {{.*}}%[[PTR]]
