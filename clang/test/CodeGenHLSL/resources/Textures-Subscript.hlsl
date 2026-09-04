// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DREG0=t0 -DREG1=t1 -DREG2=t2 -DTEXTURE=Texture2D -DCOORD_TYPE=uint2 \
// RUN:   -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL -DROV_OR_COUNT=0 \
// RUN:   -DDXIL_HANDLE=dx.Texture -DDXIL_TY=2 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DREG0=t0 -DREG1=t1 -DREG2=t2 -DTEXTURE=Texture2D -DCOORD_TYPE=uint2 \
// RUN:   -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,SPIRV -DARRAYED=0 -DMS=0 -DSAMPLED=1 \
// RUN:   -DFMT_FLOAT4=0 -DFMT_FLOAT=0 -DFMT_INT3=0 -DSPV_DIM=1
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DREG0=t0 -DREG1=t1 -DREG2=t2 -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=uint3 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,DXIL -DROV_OR_COUNT=0 \
// RUN:   -DDXIL_HANDLE=dx.Texture -DDXIL_TY=7 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DREG0=t0 -DREG1=t1 -DREG2=t2 -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=uint3 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,SPIRV -DARRAYED=1 -DMS=0 -DSAMPLED=1 \
// RUN:   -DFMT_FLOAT4=0 -DFMT_FLOAT=0 -DFMT_INT3=0 -DSPV_DIM=1
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DREG0=t0 -DREG1=t1 -DREG2=t2 -DTEXTURE=Texture2DMS \
// RUN:   -DCOORD_TYPE=uint2 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DMS -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,DXIL -DROV_OR_COUNT=0 \
// RUN:   -DDXIL_HANDLE=dx.MSTexture -DDXIL_TY=3 -DRW=0
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DREG0=t0 -DREG1=t1 -DREG2=t2 -DTEXTURE=Texture2DMS \
// RUN:   -DCOORD_TYPE=uint2 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DMS -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,SPIRV -DARRAYED=0 -DMS=1 -DSAMPLED=1 \
// RUN:   -DFMT_FLOAT4=0 -DFMT_FLOAT=0 -DFMT_INT3=0 -DSPV_DIM=1
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DHAS_STORE -DREG0=u0 -DREG1=u1 -DREG2=u2 -DTEXTURE=RWTexture2D \
// RUN:   -DCOORD_TYPE=uint2 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=RWTexture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,CHECK-STORE,DXIL -DRW=1 -DROV_OR_COUNT=0 \
// RUN:   -DDXIL_HANDLE=dx.Texture -DDXIL_TY=2
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DHAS_STORE -DREG0=u0 -DREG1=u1 -DREG2=u2 -DTEXTURE=RWTexture2D \
// RUN:   -DCOORD_TYPE=uint2 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=RWTexture2D -DCOORD_DIM=2 \
// RUN:   --check-prefixes=CHECK,CHECK-STORE,SPIRV -DARRAYED=0 -DMS=0 \
// RUN:   -DSAMPLED=2 -DFMT_FLOAT4=1 -DFMT_FLOAT=3 -DFMT_INT3=0 -DSPV_DIM=1
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DHAS_STORE -DREG0=u0 -DREG1=u1 -DREG2=u2 -DTEXTURE=RWTexture2DArray \
// RUN:   -DCOORD_TYPE=uint3 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=RWTexture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,CHECK-STORE,DXIL -DRW=1 -DROV_OR_COUNT=0 \
// RUN:   -DDXIL_HANDLE=dx.Texture -DDXIL_TY=7
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -Wno-sign-conversion \
// RUN:   -DHAS_STORE -DREG0=u0 -DREG1=u1 -DREG2=u2 -DTEXTURE=RWTexture2DArray \
// RUN:   -DCOORD_TYPE=uint3 -o - %s \
// RUN:   | llvm-cxxfilt \
// RUN:   | FileCheck %s -DTEXTURE=RWTexture2DArray -DCOORD_DIM=3 \
// RUN:   --check-prefixes=CHECK,CHECK-STORE,SPIRV -DARRAYED=1 -DMS=0 \
// RUN:   -DSAMPLED=2 -DFMT_FLOAT4=1 -DFMT_FLOAT=3 -DFMT_INT3=0 -DSPV_DIM=1

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   REG0               register binding for Tex (t# for read-only, u# for
//                      writable)
//   REG1               register binding for Tex2
//   REG2               register binding for Tex3
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//   COORD_DIM          sample location components (DIM plus the array slice)
//   ROV_OR_COUNT       the overloaded second dx handle operand: IsROV for
//                      dx.Texture, the sample count for dx.MSTexture
//   DXIL_HANDLE        DXIL resource handle type (dx.Texture or dx.MSTexture)
//   DXIL_TY            dx.Texture resource-kind operand
//   RW                 dx.Texture UAV operand
//   ARRAYED            spirv.Image Arrayed operand
//   MS                 spirv.Image MS (multisampled) operand
//   SAMPLED            spirv.Image Sampled operand
//   FMT_FLOAT4         spirv.Image Image Format for the float4 texture;
//                      uniform for sampled images but per-element for storage
//                      images
//   FMT_FLOAT          spirv.Image Image Format for the float texture
//   FMT_INT3           spirv.Image Image Format for the int3 texture
//   SPV_DIM            spirv.Image Dim operand
//   HAS_STORE          defined for writable (UAV) textures, which can
//                      additionally store through operator[]
//
// Check prefixes:
//   STORE              storing through operator[]
//
// Texture2DMS reuses the same operator[] codegen; only the resource handle type
// differs (dx.MSTexture / a multisampled spirv.Image). It reads sample 0.
// ROV_OR_COUNT is the overloaded second int operand of the DXIL handle type:
// IsROV for dx.Texture, the sample count for dx.MSTexture. It is 0 for every
// texture below (none are ROVs, and Texture2DMS<T> defaults to a runtime
// sample count), but a ROV or an explicit Texture2DMS<T, N> would need its own
// value here.

TEXTURE<float4> Tex : register(REG0);
TEXTURE<float> Tex2 : register(REG1);
TEXTURE<int3> Tex3 : register(REG2);

[numthreads(1,1,1)]
void main(COORD_TYPE DTid : SV_DispatchThreadID) {
  // Every texture type can read through the subscript.
  float4 val = Tex[DTid];
  float val2 = Tex2[DTid];
  int3 val3 = Tex3[DTid];

#ifdef HAS_STORE
  // A writable texture can also store through it.
  Tex[DTid] = float4(1, 2, 3, 4);
  Tex[DTid].y = 5.0;
  Tex2[DTid] = 6.0;
  Tex3[DTid] = int3(7, 8, 9);
#endif
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

// Store a whole texel.
// CHECK-STORE: %[[DTID_VAL:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[DTID_ADDR]]
// CHECK-STORE: %[[CALL1:.*]] = call {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<float vector[4]{{(, [0-9]+)?}}>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr {{.*}} @Tex, <[[COORD_DIM]] x i32> noundef %[[DTID_VAL]])
// CHECK-STORE: store <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>, ptr{{.*}} %[[CALL1]]

// Store a single component: a GEP off the texel pointer, which the DXIL
// backend later expands into a read-modify-write of the whole texel.
// CHECK-STORE: %[[DTID_VAL2:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[DTID_ADDR]]
// CHECK-STORE: %[[CALL2:.*]] = call {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<float vector[4]{{(, [0-9]+)?}}>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr {{.*}} @Tex, <[[COORD_DIM]] x i32> noundef %[[DTID_VAL2]])
// CHECK-STORE: %[[ELEM:.*]] = getelementptr <4 x float>, ptr{{.*}} %[[CALL2]], i32 0, i32 1
// CHECK-STORE: store float 5.000000e+00, ptr{{.*}} %[[ELEM]]

// Store to a scalar texture.
// CHECK-STORE: %[[DTID_VAL3:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[DTID_ADDR]]
// CHECK-STORE: %[[CALL3:.*]] = call {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<float{{(, [0-9]+)?}}>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr {{.*}} @Tex2, <[[COORD_DIM]] x i32> noundef %[[DTID_VAL3]])
// CHECK-STORE: store float 6.000000e+00, ptr{{.*}} %[[CALL3]]

// Store to an integer texture.
// CHECK-STORE: %[[DTID_VAL4:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[DTID_ADDR]]
// CHECK-STORE: %[[CALL4:.*]] = call {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<int vector[3]{{(, [0-9]+)?}}>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr {{.*}} @Tex3, <[[COORD_DIM]] x i32> noundef %[[DTID_VAL4]])
// CHECK-STORE: store <3 x i32> <i32 7, i32 8, i32 9>, ptr{{.*}} %[[CALL4]]

// CHECK: define linkonce_odr hidden {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<float vector[4]{{(, [0-9]+)?}}>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %[[THIS:.*]], <[[COORD_DIM]] x i32> noundef %[[INDEX:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[INDEX_ADDR:.*]] = alloca <[[COORD_DIM]] x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x i32> %[[INDEX]], ptr %[[INDEX_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr {{.*}} %"class.hlsl::[[TEXTURE]]", ptr %[[THIS1]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("[[DXIL_HANDLE]]", <4 x float>, [[RW]], [[ROV_OR_COUNT]], 0, [[DXIL_TY]]), ptr %[[HANDLE_PTR]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], [[MS]], [[SAMPLED]], [[FMT_FLOAT4]]), ptr %[[HANDLE_PTR]]
// CHECK: %[[INDEX_VAL:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[INDEX_ADDR]]
// DXIL: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.{{.*}}(target("[[DXIL_HANDLE]]", <4 x float>, [[RW]], [[ROV_OR_COUNT]], 0, [[DXIL_TY]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// SPIRV: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], [[MS]], [[SAMPLED]], [[FMT_FLOAT4]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// CHECK: ret ptr {{.*}}%[[PTR]]

// CHECK: define linkonce_odr hidden {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<float{{(, [0-9]+)?}}>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %[[THIS:.*]], <[[COORD_DIM]] x i32> noundef %[[INDEX:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[INDEX_ADDR:.*]] = alloca <[[COORD_DIM]] x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x i32> %[[INDEX]], ptr %[[INDEX_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr {{.*}} %"class.hlsl::[[TEXTURE]].0", ptr %[[THIS1]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("[[DXIL_HANDLE]]", float, [[RW]], [[ROV_OR_COUNT]], 0, [[DXIL_TY]]), ptr %[[HANDLE_PTR]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], [[MS]], [[SAMPLED]], [[FMT_FLOAT]]), ptr %[[HANDLE_PTR]]
// CHECK: %[[INDEX_VAL:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[INDEX_ADDR]]
// DXIL: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.{{.*}}(target("[[DXIL_HANDLE]]", float, [[RW]], [[ROV_OR_COUNT]], 0, [[DXIL_TY]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// SPIRV: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], [[MS]], [[SAMPLED]], [[FMT_FLOAT]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// CHECK: ret ptr {{.*}}%[[PTR]]

// CHECK: define linkonce_odr hidden {{(spir_func )?}}noundef {{.*}}ptr{{.*}} @hlsl::[[TEXTURE]]<int vector[3]{{(, [0-9]+)?}}>::operator[](unsigned int vector[[[COORD_DIM]]]) const(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %[[THIS:.*]], <[[COORD_DIM]] x i32> noundef %[[INDEX:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[INDEX_ADDR:.*]] = alloca <[[COORD_DIM]] x i32>
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store <[[COORD_DIM]] x i32> %[[INDEX]], ptr %[[INDEX_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr {{.*}} %"class.hlsl::[[TEXTURE]].1", ptr %[[THIS1]], i32 0, i32 0
// DXIL: %[[HANDLE:.*]] = load target("[[DXIL_HANDLE]]", <3 x i32>, [[RW]], [[ROV_OR_COUNT]], 1, [[DXIL_TY]]), ptr %[[HANDLE_PTR]]
// SPIRV: %[[HANDLE:.*]] = load target("spirv.SignedImage", i32, [[SPV_DIM]], 2, [[ARRAYED]], [[MS]], [[SAMPLED]], [[FMT_INT3]]), ptr %[[HANDLE_PTR]]
// CHECK: %[[INDEX_VAL:.*]] = load <[[COORD_DIM]] x i32>, ptr %[[INDEX_ADDR]]
// DXIL: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.{{.*}}(target("[[DXIL_HANDLE]]", <3 x i32>, [[RW]], [[ROV_OR_COUNT]], 1, [[DXIL_TY]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// SPIRV: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{{.*}}(target("spirv.SignedImage", i32, [[SPV_DIM]], 2, [[ARRAYED]], [[MS]], [[SAMPLED]], [[FMT_INT3]]) %[[HANDLE]], <[[COORD_DIM]] x i32> %[[INDEX_VAL]])
// CHECK: ret ptr {{.*}}%[[PTR]]
