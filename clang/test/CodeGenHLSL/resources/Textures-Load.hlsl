// Texture1D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int \
// RUN:   -DHAS_OFFSET -DOFFSET_ARG="1" -DTEXTURE=Texture1D \
// RUN:   -DLOAD_ARG="int2(loc, 0)" -o - %s | llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SRV,SCALAR-COORD,WIDE-LOC,DXIL,DXIL-SRV \
// RUN:   -DTEXTURE=Texture1D -D#LOAD_DIM=2 -DCOORD_DIM=1 -DDXIL_TY=1 -DRW=0 \
// RUN:   -DENTRY_DIM=1 -DDIM=1 -DCOORD_LLVM=i32 -DOFFSET_LLVM=i32 \
// RUN:   -DOFFSET_CXX=int -DOFFSET_ZERO=0 -DOFFSET_CONST=1 \
// RUN:   -DENTRY_CXX=int -DLOAD_LLVM="<2 x i32>" \
// RUN:   -DLOAD_CXX="int vector[2]"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int \
// RUN:   -DHAS_OFFSET -DOFFSET_ARG="1" -DTEXTURE=Texture1D \
// RUN:   -DLOAD_ARG="int2(loc, 0)" -o - %s | llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SRV,SCALAR-COORD,WIDE-LOC,SPIRV,SPIRV-SRV \
// RUN:   -DTEXTURE=Texture1D -D#LOAD_DIM=2 -DCOORD_DIM=1 -DARRAYED=0 \
// RUN:   -DSAMPLED=1 -DFORMAT1=0 -DFORMAT3=0 -DFORMAT6=0 -DFORMAT21=0 \
// RUN:   -DFORMAT24=0 -DFORMAT25=0 -DSPV_DIM=0 -DENTRY_DIM=1 -DDIM=1 \
// RUN:   -DCOORD_LLVM=i32 -DOFFSET_LLVM=i32 -DOFFSET_CXX=int \
// RUN:   -DOFFSET_ZERO=0 -DOFFSET_CONST=1 -DENTRY_CXX=int \
// RUN:   -DLOAD_LLVM="<2 x i32>" -DLOAD_CXX="int vector[2]"

// Texture1DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DHAS_OFFSET -DOFFSET_ARG="1" -DTEXTURE=Texture1DArray \
// RUN:   -DLOAD_ARG="int3(loc, 0)" -o - %s | llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SRV,VEC-COORD,WIDE-LOC,DXIL,DXIL-SRV \
// RUN:   -DTEXTURE=Texture1DArray -D#LOAD_DIM=3 -DCOORD_DIM=2 \
// RUN:   -DCOORD_MASK="<i32 0, i32 1>" -DDXIL_TY=6 -DRW=0 -DENTRY_DIM=2 \
// RUN:   -DDIM=1 -DCOORD_LLVM="<2 x i32>" -DOFFSET_LLVM=i32 \
// RUN:   -DOFFSET_CXX=int -DOFFSET_ZERO=0 -DOFFSET_CONST=1 \
// RUN:   -DENTRY_CXX="int vector[2]" -DLOAD_LLVM="<3 x i32>" \
// RUN:   -DLOAD_CXX="int vector[3]"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DHAS_OFFSET -DOFFSET_ARG="1" -DTEXTURE=Texture1DArray \
// RUN:   -DLOAD_ARG="int3(loc, 0)" -o - %s | llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SRV,VEC-COORD,WIDE-LOC,SPIRV,SPIRV-SRV \
// RUN:   -DTEXTURE=Texture1DArray -D#LOAD_DIM=3 -DCOORD_DIM=2 \
// RUN:   -DCOORD_MASK="<i32 0, i32 1>" -DARRAYED=1 -DSAMPLED=1 -DFORMAT1=0 \
// RUN:   -DFORMAT3=0 -DFORMAT6=0 -DFORMAT21=0 -DFORMAT24=0 -DFORMAT25=0 \
// RUN:   -DSPV_DIM=0 -DENTRY_DIM=2 -DDIM=1 -DCOORD_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_LLVM=i32 -DOFFSET_CXX=int -DOFFSET_ZERO=0 \
// RUN:   -DOFFSET_CONST=1 -DENTRY_CXX="int vector[2]" \
// RUN:   -DLOAD_LLVM="<3 x i32>" -DLOAD_CXX="int vector[3]"
// Texture2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DHAS_OFFSET -DOFFSET_ARG="int2(1, 1)" -DTEXTURE=Texture2D \
// RUN:   -DLOAD_ARG="int3(loc, 0)" -o - %s | llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SRV,VEC-COORD,WIDE-LOC,DXIL,DXIL-SRV \
// RUN:   -DTEXTURE=Texture2D -D#LOAD_DIM=3 -DCOORD_DIM=2 \
// RUN:   -DCOORD_MASK="<i32 0, i32 1>" -DDXIL_TY=2 -DRW=0 -DENTRY_DIM=2 \
// RUN:   -DDIM=2 -DCOORD_LLVM="<2 x i32>" -DOFFSET_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[2]" -DOFFSET_ZERO=zeroinitializer \
// RUN:   -DOFFSET_CONST="splat (i32 1)" -DENTRY_CXX="int vector[2]" \
// RUN:   -DLOAD_LLVM="<3 x i32>" -DLOAD_CXX="int vector[3]"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DHAS_OFFSET -DOFFSET_ARG="int2(1, 1)" -DTEXTURE=Texture2D \
// RUN:   -DLOAD_ARG="int3(loc, 0)" -o - %s | llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SRV,VEC-COORD,WIDE-LOC,SPIRV,SPIRV-SRV \
// RUN:   -DTEXTURE=Texture2D -D#LOAD_DIM=3 -DCOORD_DIM=2 \
// RUN:   -DCOORD_MASK="<i32 0, i32 1>" -DARRAYED=0 -DSAMPLED=1 -DFORMAT1=0 \
// RUN:   -DFORMAT3=0 -DFORMAT6=0 -DFORMAT21=0 -DFORMAT24=0 -DFORMAT25=0 \
// RUN:   -DSPV_DIM=1 -DENTRY_DIM=2 -DDIM=2 -DCOORD_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_LLVM="<2 x i32>" -DOFFSET_CXX="int vector[2]" \
// RUN:   -DOFFSET_ZERO=zeroinitializer -DOFFSET_CONST="splat (i32 1)" \
// RUN:   -DENTRY_CXX="int vector[2]" -DLOAD_LLVM="<3 x i32>" \
// RUN:   -DLOAD_CXX="int vector[3]"

// Texture2DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DHAS_OFFSET -DOFFSET_ARG="int2(1, 1)" -DTEXTURE=Texture2DArray \
// RUN:   -DLOAD_ARG="int4(loc, 0, 0)" -o - %s | llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SRV,VEC-COORD,WIDE-LOC,DXIL,DXIL-SRV \
// RUN:   -DTEXTURE=Texture2DArray -D#LOAD_DIM=4 -DCOORD_DIM=3 \
// RUN:   -DCOORD_MASK="<i32 0, i32 1, i32 2>" -DDXIL_TY=7 -DRW=0 \
// RUN:   -DENTRY_DIM=2 -DDIM=2 -DCOORD_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_LLVM="<2 x i32>" -DOFFSET_CXX="int vector[2]" \
// RUN:   -DOFFSET_ZERO=zeroinitializer -DOFFSET_CONST="splat (i32 1)" \
// RUN:   -DENTRY_CXX="int vector[2]" -DLOAD_LLVM="<4 x i32>" \
// RUN:   -DLOAD_CXX="int vector[4]"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DHAS_OFFSET -DOFFSET_ARG="int2(1, 1)" -DTEXTURE=Texture2DArray \
// RUN:   -DLOAD_ARG="int4(loc, 0, 0)" -o - %s | llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SRV,VEC-COORD,WIDE-LOC,SPIRV,SPIRV-SRV \
// RUN:   -DTEXTURE=Texture2DArray -D#LOAD_DIM=4 -DCOORD_DIM=3 \
// RUN:   -DCOORD_MASK="<i32 0, i32 1, i32 2>" -DARRAYED=1 -DSAMPLED=1 \
// RUN:   -DFORMAT1=0 -DFORMAT3=0 -DFORMAT6=0 -DFORMAT21=0 -DFORMAT24=0 \
// RUN:   -DFORMAT25=0 -DSPV_DIM=1 -DENTRY_DIM=2 -DDIM=2 \
// RUN:   -DCOORD_LLVM="<3 x i32>" -DOFFSET_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[2]" -DOFFSET_ZERO=zeroinitializer \
// RUN:   -DOFFSET_CONST="splat (i32 1)" -DENTRY_CXX="int vector[2]" \
// RUN:   -DLOAD_LLVM="<4 x i32>" -DLOAD_CXX="int vector[4]"

// RWTexture1D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int \
// RUN:   -DTEXTURE=RWTexture1D -DLOAD_ARG="loc" -o - %s | llvm-cxxfilt | \
// RUN:   FileCheck %s --check-prefixes=CHECK,UAV,EXACT-LOC,DXIL,DXIL-UAV \
// RUN:   -DTEXTURE=RWTexture1D -D#LOAD_DIM=1 -DCOORD_DIM=1 -DDXIL_TY=1 -DRW=1 \
// RUN:   -DENTRY_DIM=1 -DDIM=1 -DCOORD_LLVM=i32 -DOFFSET_LLVM=i32 \
// RUN:   -DOFFSET_CXX=int -DOFFSET_ZERO=0 -DOFFSET_CONST=1 \
// RUN:   -DENTRY_CXX=int -DLOAD_LLVM=i32 -DLOAD_CXX=int
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int \
// RUN:   -DTEXTURE=RWTexture1D -DLOAD_ARG="loc" -o - %s | llvm-cxxfilt | \
// RUN:   FileCheck %s --check-prefixes=CHECK,UAV,EXACT-LOC,SPIRV,SPIRV-UAV \
// RUN:   -DTEXTURE=RWTexture1D -D#LOAD_DIM=1 -DCOORD_DIM=1 -DARRAYED=0 \
// RUN:   -DSAMPLED=2 -DFORMAT1=1 -DFORMAT3=3 -DFORMAT6=6 -DFORMAT21=21 \
// RUN:   -DFORMAT24=24 -DFORMAT25=25 -DSPV_DIM=0 -DENTRY_DIM=1 -DDIM=1 \
// RUN:   -DCOORD_LLVM=i32 -DOFFSET_LLVM=i32 -DOFFSET_CXX=int \
// RUN:   -DOFFSET_ZERO=0 -DOFFSET_CONST=1 -DENTRY_CXX=int -DLOAD_LLVM=i32 \
// RUN:   -DLOAD_CXX=int
// RWTexture2D
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DTEXTURE=RWTexture2D -DLOAD_ARG="loc" -o - %s | llvm-cxxfilt | \
// RUN:   FileCheck %s --check-prefixes=CHECK,UAV,EXACT-LOC,DXIL,DXIL-UAV \
// RUN:   -DTEXTURE=RWTexture2D -D#LOAD_DIM=2 -DCOORD_DIM=2 -DDXIL_TY=2 -DRW=1 \
// RUN:   -DENTRY_DIM=2 -DDIM=2 -DCOORD_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_LLVM="<2 x i32>" -DOFFSET_CXX="int vector[2]" \
// RUN:   -DOFFSET_ZERO=zeroinitializer -DOFFSET_CONST="splat (i32 1)" \
// RUN:   -DENTRY_CXX="int vector[2]" -DLOAD_LLVM="<2 x i32>" \
// RUN:   -DLOAD_CXX="int vector[2]"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DTEXTURE=RWTexture2D -DLOAD_ARG="loc" -o - %s | llvm-cxxfilt | \
// RUN:   FileCheck %s --check-prefixes=CHECK,UAV,EXACT-LOC,SPIRV,SPIRV-UAV \
// RUN:   -DTEXTURE=RWTexture2D -D#LOAD_DIM=2 -DCOORD_DIM=2 -DARRAYED=0 \
// RUN:   -DSAMPLED=2 -DFORMAT1=1 -DFORMAT3=3 -DFORMAT6=6 -DFORMAT21=21 \
// RUN:   -DFORMAT24=24 -DFORMAT25=25 -DSPV_DIM=1 -DENTRY_DIM=2 -DDIM=2 \
// RUN:   -DCOORD_LLVM="<2 x i32>" -DOFFSET_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[2]" -DOFFSET_ZERO=zeroinitializer \
// RUN:   -DOFFSET_CONST="splat (i32 1)" -DENTRY_CXX="int vector[2]" \
// RUN:   -DLOAD_LLVM="<2 x i32>" -DLOAD_CXX="int vector[2]"

// RWTexture1DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int \
// RUN:   -DTEXTURE=RWTexture1DArray -DLOAD_ARG="int2(loc, 0)" -o - %s | \
// RUN:   llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,UAV,WIDE-LOC,DXIL,DXIL-UAV \
// RUN:   -DTEXTURE=RWTexture1DArray -D#LOAD_DIM=2 -DCOORD_DIM=2 -DDXIL_TY=6 \
// RUN:   -DRW=1 -DENTRY_DIM=1 -DDIM=1 -DCOORD_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_LLVM=i32 -DOFFSET_CXX=int -DOFFSET_ZERO=0 \
// RUN:   -DOFFSET_CONST=1 -DENTRY_CXX=int -DLOAD_LLVM="<2 x i32>" \
// RUN:   -DLOAD_CXX="int vector[2]"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int \
// RUN:   -DTEXTURE=RWTexture1DArray -DLOAD_ARG="int2(loc, 0)" -o - %s | \
// RUN:   llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,UAV,WIDE-LOC,SPIRV,SPIRV-UAV \
// RUN:   -DTEXTURE=RWTexture1DArray -D#LOAD_DIM=2 -DCOORD_DIM=2 -DARRAYED=1 \
// RUN:   -DSAMPLED=2 -DFORMAT1=1 -DFORMAT3=3 -DFORMAT6=6 -DFORMAT21=21 \
// RUN:   -DFORMAT24=24 -DFORMAT25=25 -DSPV_DIM=0 -DENTRY_DIM=1 -DDIM=1 \
// RUN:   -DCOORD_LLVM="<2 x i32>" -DOFFSET_LLVM=i32 -DOFFSET_CXX=int \
// RUN:   -DOFFSET_ZERO=0 -DOFFSET_CONST=1 -DENTRY_CXX=int \
// RUN:   -DLOAD_LLVM="<2 x i32>" -DLOAD_CXX="int vector[2]"
// RWTexture2DArray
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DTEXTURE=RWTexture2DArray -DLOAD_ARG="int3(loc, 0)" -o - %s | \
// RUN:   llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,UAV,WIDE-LOC,DXIL,DXIL-UAV \
// RUN:   -DTEXTURE=RWTexture2DArray -D#LOAD_DIM=3 -DCOORD_DIM=3 -DDXIL_TY=7 \
// RUN:   -DRW=1 -DENTRY_DIM=2 -DDIM=2 -DCOORD_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_LLVM="<2 x i32>" -DOFFSET_CXX="int vector[2]" \
// RUN:   -DOFFSET_ZERO=zeroinitializer -DOFFSET_CONST="splat (i32 1)" \
// RUN:   -DENTRY_CXX="int vector[2]" -DLOAD_LLVM="<3 x i32>" \
// RUN:   -DLOAD_CXX="int vector[3]"
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -DENTRY_TYPE=int2 \
// RUN:   -DTEXTURE=RWTexture2DArray -DLOAD_ARG="int3(loc, 0)" -o - %s | \
// RUN:   llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,UAV,WIDE-LOC,SPIRV,SPIRV-UAV \
// RUN:   -DTEXTURE=RWTexture2DArray -D#LOAD_DIM=3 -DCOORD_DIM=3 -DARRAYED=1 \
// RUN:   -DSAMPLED=2 -DFORMAT1=1 -DFORMAT3=3 -DFORMAT6=6 -DFORMAT21=21 \
// RUN:   -DFORMAT24=24 -DFORMAT25=25 -DSPV_DIM=1 -DENTRY_DIM=2 -DDIM=2 \
// RUN:   -DCOORD_LLVM="<3 x i32>" -DOFFSET_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[2]" -DOFFSET_ZERO=zeroinitializer \
// RUN:   -DOFFSET_CONST="splat (i32 1)" -DENTRY_CXX="int vector[2]" \
// RUN:   -DLOAD_LLVM="<3 x i32>" -DLOAD_CXX="int vector[3]"

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   ENTRY_TYPE         the entry point's own coordinate type
//   HAS_OFFSET         defined for types whose Load has an overload taking an
//                      offset
//   OFFSET_ARG         a literal offset argument
//   TEXTURE            resource type name
//   LOAD_ARG           the Load location, built from the entry point's `loc`
//   LOAD_DIM           Load location components; a FileCheck numeric variable,
//                      so the last component's index is [[#LOAD_DIM-1]]
//   LOAD_LLVM          Load location type in the IR
//   LOAD_CXX           Load location type in the C++ signature
//   COORD_DIM          sample location components (DIM plus the array slice)
//   COORD_MASK         shufflevector mask extracting the coordinate from a
//                      location; only needed where the coordinate is a vector
//   COORD_LLVM         coordinate type in the IR
//   OFFSET_LLVM        offset type in the IR
//   OFFSET_CXX         offset type in the C++ signature
//   OFFSET_ZERO        the all-zero offset as it appears in the IR
//   OFFSET_CONST       OFFSET_ARG as it appears in the IR
//   ENTRY_CXX          the entry point's coordinate type in the C++ signature
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
//
// Check prefixes:
//   SRV                read-only textures. Their location carries a trailing
//                      mip level, which is split back out of the location, and
//                      they have the offset overload.
//   UAV                writable textures. A UAV descriptor binds a single mip
//                      slice, so the location is all coordinate and the level
//                      operand is a placeholder that the backends discard.
//   WIDE-LOC           types whose Load location is wider than the entry
//                      point's coordinate, so LOAD_ARG appends the array slice
//                      and/or the mip level to `loc`
//   EXACT-LOC          types whose Load location is exactly `loc`
//   VEC-COORD          types whose coordinate is a vector, split out of the
//                      location with a shufflevector
//   SCALAR-COORD       1D types, whose coordinate is a single element

TEXTURE<float4> t;

// CHECK: define hidden {{.*}} <4 x float> @test_load([[ENTRY_CXX]])
// WIDE-LOC: %[[COORD:.*]] = insertelement [[LOAD_LLVM]] {{.*}}, i32 0, i32 [[#LOAD_DIM-1]]
// WIDE-LOC: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Load([[LOAD_CXX]])(ptr {{.*}} @t, [[LOAD_LLVM]] noundef %[[COORD]])
// EXACT-LOC: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Load([[LOAD_CXX]])(ptr {{.*}} @t, [[LOAD_LLVM]] noundef %{{.*}})
// CHECK: ret <4 x float> %[[CALL]]

float4 test_load(ENTRY_TYPE loc : LOC) : SV_Target {
  return t.Load(LOAD_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Load([[LOAD_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[LOAD_ADDR:.*]] = alloca [[LOAD_LLVM]]
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// CHECK: store [[LOAD_LLVM]] %[[LOAD]], ptr %[[LOAD_ADDR]]
// CHECK: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("{{(dx.Texture|spirv.Image)}}", {{.*}}), ptr %[[HANDLE_GEP]]
// CHECK: %[[LOAD_VAL:.*]] = load [[LOAD_LLVM]], ptr %[[LOAD_ADDR]]
// UAV-NOT: shufflevector
// UAV-NOT: extractelement
// VEC-COORD: %[[COORD:.*]] = shufflevector [[LOAD_LLVM]] %[[LOAD_VAL]], [[LOAD_LLVM]] poison, [[COORD_LLVM]] [[COORD_MASK]]
// SCALAR-COORD: %[[COORD:.*]] = extractelement [[LOAD_LLVM]] %[[LOAD_VAL]], i64 0
// SRV: %[[LOD:.*]] = extractelement [[LOAD_LLVM]] %[[LOAD_VAL]], i64 [[COORD_DIM]]
// DXIL-SRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], [[COORD_LLVM]] %[[COORD]], i32 %[[LOD]], [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// DXIL-UAV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_{{.*}}(target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], [[COORD_LLVM]] %[[LOAD_VAL]], i32 poison, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// SPIRV-SRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT1]]) %[[HANDLE]], [[COORD_LLVM]] %[[COORD]], i32 %[[LOD]], [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// SPIRV-UAV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_{{.*}}(target("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT1]]) %[[HANDLE]], [[COORD_LLVM]] %[[LOAD_VAL]], i32 poison, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// CHECK: ret <4 x float> %[[RES]]

// SRV: define hidden {{.*}} <4 x float> @test_load_offset([[ENTRY_CXX]])
// SRV: %[[COORD:.*]] = insertelement [[LOAD_LLVM]] {{.*}}, i32 0, i32 [[#LOAD_DIM-1]]
// SRV: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} @t, [[LOAD_LLVM]] noundef %[[COORD]], [[OFFSET_LLVM]] noundef [[OFFSET_CONST]])
// SRV: ret <4 x float> %[[CALL]]

#ifdef HAS_OFFSET
float4 test_load_offset(ENTRY_TYPE loc : LOC) : SV_Target {
  return t.Load(LOAD_ARG, OFFSET_ARG);
}
#endif

// SRV: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]], [[OFFSET_LLVM]] {{.*}} %[[OFFSET:.*]])
// SRV: %[[THIS_ADDR:.*]] = alloca ptr
// SRV: %[[LOAD_ADDR:.*]] = alloca [[LOAD_LLVM]]
// SRV: %[[OFFSET_ADDR:.*]] = alloca [[OFFSET_LLVM]]
// SRV: store ptr %[[THIS]], ptr %[[THIS_ADDR]]
// SRV: store [[LOAD_LLVM]] %[[LOAD]], ptr %[[LOAD_ADDR]]
// SRV: store [[OFFSET_LLVM]] %[[OFFSET]], ptr %[[OFFSET_ADDR]]
// SRV: %[[THIS_VAL:.*]] = load ptr, ptr %[[THIS_ADDR]]
// SRV: %[[HANDLE_GEP:.*]] = getelementptr inbounds nuw %"class.hlsl::[[TEXTURE]]", ptr %[[THIS_VAL]], i32 0, i32 0
// SRV: %[[HANDLE:.*]] = load target("{{(dx.Texture|spirv.Image)}}", {{.*}}), ptr %[[HANDLE_GEP]]
// SRV: %[[LOAD_VAL:.*]] = load [[LOAD_LLVM]], ptr %[[LOAD_ADDR]]
// VEC-COORD: %[[COORD:.*]] = shufflevector [[LOAD_LLVM]] %[[LOAD_VAL]], [[LOAD_LLVM]] poison, [[COORD_LLVM]] [[COORD_MASK]]
// SCALAR-COORD: %[[COORD:.*]] = extractelement [[LOAD_LLVM]] %[[LOAD_VAL]], i64 0
// SRV: %[[LOD:.*]] = extractelement [[LOAD_LLVM]] %[[LOAD_VAL]], i64 [[COORD_DIM]]
// SRV: %[[OFFSET_VAL:.*]] = load [[OFFSET_LLVM]], ptr %[[OFFSET_ADDR]]
// DXIL-SRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_{{.*}}("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %[[HANDLE]], [[COORD_LLVM]] %[[COORD]], i32 %[[LOD]], [[OFFSET_LLVM]] %[[OFFSET_VAL]])
// SPIRV-SRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT1]]) %[[HANDLE]], [[COORD_LLVM]] %[[COORD]], i32 %[[LOD]], [[OFFSET_LLVM]] %[[OFFSET_VAL]])
// SRV: ret <4 x float> %[[RES]]


// For the rest of the types, we just check that the call to the member
// function has the correct return type.

TEXTURE<float> t_float;

// CHECK: define hidden {{.*}} float @test_load_float([[ENTRY_CXX]])
// CHECK: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float>::Load([[LOAD_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.load.level.f32.tdx.Texture_f32_{{.*}}("dx.Texture", float, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.load.level.f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT3]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// CHECK: ret float %[[RES]]
float test_load_float(ENTRY_TYPE loc : LOC) {
  return t_float.Load(LOAD_ARG);
}

#ifdef HAS_OFFSET
// SRV: define hidden {{.*}} float @test_load_offset_float([[ENTRY_CXX]])
// SRV: %[[CALL:.*]] = call {{.*}} float @hlsl::[[TEXTURE]]<float>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} @t_float, [[LOAD_LLVM]] noundef %{{.*}}, [[OFFSET_LLVM]] noundef [[OFFSET_CONST]])
// SRV: ret float %[[CALL]]
float test_load_offset_float(ENTRY_TYPE loc : LOC) {
  return t_float.Load(LOAD_ARG, OFFSET_ARG);
}
#endif

// SRV: define linkonce_odr hidden {{.*}} float @hlsl::[[TEXTURE]]<float>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]], [[OFFSET_LLVM]] {{.*}} %[[OFFSET:.*]])
// DXIL-SRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.dx.resource.load.level.f32.tdx.Texture_f32_{{.*}}("dx.Texture", float, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SPIRV-SRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn float @llvm.spv.resource.load.level.f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT3]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SRV: ret float %[[RES]]

TEXTURE<float2> t_float2;

// CHECK: define hidden {{.*}} <2 x float> @test_load_float2([[ENTRY_CXX]])
// CHECK: %[[CALL:.*]] = call {{.*}} <2 x float> @hlsl::[[TEXTURE]]<float vector[2]>::Load([[LOAD_CXX]])(ptr {{.*}} @t_float2, [[LOAD_LLVM]] noundef %{{.*}})
// CHECK: ret <2 x float> %[[CALL]]
float2 test_load_float2(ENTRY_TYPE loc : LOC) {
  return t_float2.Load(LOAD_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <2 x float> @hlsl::[[TEXTURE]]<float vector[2]>::Load([[LOAD_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.dx.resource.load.level.v2f32.tdx.Texture_v2f32_{{.*}}("dx.Texture", <2 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.resource.load.level.v2f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT6]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// CHECK: ret <2 x float> %[[RES]]

#ifdef HAS_OFFSET
// SRV: define hidden {{.*}} <2 x float> @test_load_offset_float2([[ENTRY_CXX]])
// SRV: %[[CALL:.*]] = call {{.*}} <2 x float> @hlsl::[[TEXTURE]]<float vector[2]>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} @t_float2, [[LOAD_LLVM]] noundef %{{.*}}, [[OFFSET_LLVM]] noundef [[OFFSET_CONST]])
// SRV: ret <2 x float> %[[CALL]]
float2 test_load_offset_float2(ENTRY_TYPE loc : LOC) {
  return t_float2.Load(LOAD_ARG, OFFSET_ARG);
}
#endif

// SRV: define linkonce_odr hidden {{.*}} <2 x float> @hlsl::[[TEXTURE]]<float vector[2]>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]], [[OFFSET_LLVM]] {{.*}} %[[OFFSET:.*]])
// DXIL-SRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.dx.resource.load.level.v2f32.tdx.Texture_v2f32_{{.*}}("dx.Texture", <2 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SPIRV-SRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.resource.load.level.v2f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT6]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SRV: ret <2 x float> %[[RES]]

TEXTURE<float3> t_float3;

// CHECK: define hidden {{.*}} <3 x float> @test_load_float3([[ENTRY_CXX]])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x float> @hlsl::[[TEXTURE]]<float vector[3]>::Load([[LOAD_CXX]])(ptr {{.*}} @t_float3, [[LOAD_LLVM]] noundef %{{.*}})
// CHECK: ret <3 x float> %[[CALL]]
float3 test_load_float3(ENTRY_TYPE loc : LOC) {
  return t_float3.Load(LOAD_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x float> @hlsl::[[TEXTURE]]<float vector[3]>::Load([[LOAD_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.dx.resource.load.level.v3f32.tdx.Texture_v3f32_{{.*}}("dx.Texture", <3 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// SPIRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.spv.resource.load.level.v3f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], 0) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// CHECK: ret <3 x float> %[[RES]]

#ifdef HAS_OFFSET
// SRV: define hidden {{.*}} <3 x float> @test_load_offset_float3([[ENTRY_CXX]])
// SRV: %[[CALL:.*]] = call {{.*}} <3 x float> @hlsl::[[TEXTURE]]<float vector[3]>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} @t_float3, [[LOAD_LLVM]] noundef %{{.*}}, [[OFFSET_LLVM]] noundef [[OFFSET_CONST]])
// SRV: ret <3 x float> %[[CALL]]
float3 test_load_offset_float3(ENTRY_TYPE loc : LOC) {
  return t_float3.Load(LOAD_ARG, OFFSET_ARG);
}
#endif

// SRV: define linkonce_odr hidden {{.*}} <3 x float> @hlsl::[[TEXTURE]]<float vector[3]>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]], [[OFFSET_LLVM]] {{.*}} %[[OFFSET:.*]])
// DXIL-SRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.dx.resource.load.level.v3f32.tdx.Texture_v3f32_{{.*}}("dx.Texture", <3 x float>, [[RW]], 0, 0, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SPIRV-SRV: %[[RES:.*]] = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.spv.resource.load.level.v3f32.tspirv.Image_f32_{{.*}}("spirv.Image", float, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], 0) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SRV: ret <3 x float> %[[RES]]

TEXTURE<int> t_int;

// CHECK: define hidden {{.*}} i32 @test_load_int([[ENTRY_CXX]])
// CHECK: %[[CALL:.*]] = call {{.*}} i32 @hlsl::[[TEXTURE]]<int>::Load([[LOAD_CXX]])(ptr {{.*}} @t_int, [[LOAD_LLVM]] noundef %{{.*}})
// CHECK: ret i32 %[[CALL]]
int test_load_int(ENTRY_TYPE loc : LOC) {
  return t_int.Load(LOAD_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} i32 @hlsl::[[TEXTURE]]<int>::Load([[LOAD_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call i32 @llvm.dx.resource.load.level.i32.tdx.Texture_i32_{{.*}}("dx.Texture", i32, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// SPIRV: %[[RES:.*]] = call i32 @llvm.spv.resource.load.level.i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT24]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// CHECK: ret i32 %[[RES]]

#ifdef HAS_OFFSET
// SRV: define hidden {{.*}} i32 @test_load_offset_int([[ENTRY_CXX]])
// SRV: %[[CALL:.*]] = call {{.*}} i32 @hlsl::[[TEXTURE]]<int>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} @t_int, [[LOAD_LLVM]] noundef %{{.*}}, [[OFFSET_LLVM]] noundef [[OFFSET_CONST]])
// SRV: ret i32 %[[CALL]]
int test_load_offset_int(ENTRY_TYPE loc : LOC) {
  return t_int.Load(LOAD_ARG, OFFSET_ARG);
}
#endif

// SRV: define linkonce_odr hidden {{.*}} i32 @hlsl::[[TEXTURE]]<int>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]], [[OFFSET_LLVM]] {{.*}} %[[OFFSET:.*]])
// DXIL-SRV: %[[RES:.*]] = call i32 @llvm.dx.resource.load.level.i32.tdx.Texture_i32_{{.*}}("dx.Texture", i32, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SPIRV-SRV: %[[RES:.*]] = call i32 @llvm.spv.resource.load.level.i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT24]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SRV: ret i32 %[[RES]]

TEXTURE<int2> t_int2;

// CHECK: define hidden {{.*}} <2 x i32> @test_load_int2([[ENTRY_CXX]])
// CHECK: %[[CALL:.*]] = call {{.*}} <2 x i32> @hlsl::[[TEXTURE]]<int vector[2]>::Load([[LOAD_CXX]])(ptr {{.*}} @t_int2, [[LOAD_LLVM]] noundef %{{.*}})
// CHECK: ret <2 x i32> %[[CALL]]
int2 test_load_int2(ENTRY_TYPE loc : LOC) {
  return t_int2.Load(LOAD_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <2 x i32> @hlsl::[[TEXTURE]]<int vector[2]>::Load([[LOAD_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call <2 x i32> @llvm.dx.resource.load.level.v2i32.tdx.Texture_v2i32_{{.*}}("dx.Texture", <2 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// SPIRV: %[[RES:.*]] = call <2 x i32> @llvm.spv.resource.load.level.v2i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT25]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// CHECK: ret <2 x i32> %[[RES]]

#ifdef HAS_OFFSET
// SRV: define hidden {{.*}} <2 x i32> @test_load_offset_int2([[ENTRY_CXX]])
// SRV: %[[CALL:.*]] = call {{.*}} <2 x i32> @hlsl::[[TEXTURE]]<int vector[2]>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} @t_int2, [[LOAD_LLVM]] noundef %{{.*}}, [[OFFSET_LLVM]] noundef [[OFFSET_CONST]])
// SRV: ret <2 x i32> %[[CALL]]
int2 test_load_offset_int2(ENTRY_TYPE loc : LOC) {
  return t_int2.Load(LOAD_ARG, OFFSET_ARG);
}
#endif

// SRV: define linkonce_odr hidden {{.*}} <2 x i32> @hlsl::[[TEXTURE]]<int vector[2]>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]], [[OFFSET_LLVM]] {{.*}} %[[OFFSET:.*]])
// DXIL-SRV: %[[RES:.*]] = call <2 x i32> @llvm.dx.resource.load.level.v2i32.tdx.Texture_v2i32_{{.*}}("dx.Texture", <2 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SPIRV-SRV: %[[RES:.*]] = call <2 x i32> @llvm.spv.resource.load.level.v2i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT25]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SRV: ret <2 x i32> %[[RES]]

TEXTURE<int3> t_int3;

// CHECK: define hidden {{.*}} <3 x i32> @test_load_int3([[ENTRY_CXX]])
// CHECK: %[[CALL:.*]] = call {{.*}} <3 x i32> @hlsl::[[TEXTURE]]<int vector[3]>::Load([[LOAD_CXX]])(ptr {{.*}} @t_int3, [[LOAD_LLVM]] noundef %{{.*}})
// CHECK: ret <3 x i32> %[[CALL]]
int3 test_load_int3(ENTRY_TYPE loc : LOC) {
  return t_int3.Load(LOAD_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <3 x i32> @hlsl::[[TEXTURE]]<int vector[3]>::Load([[LOAD_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.load.level.v3i32.tdx.Texture_v3i32_{{.*}}("dx.Texture", <3 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// SPIRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.load.level.v3i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], 0) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// CHECK: ret <3 x i32> %[[RES]]

#ifdef HAS_OFFSET
// SRV: define hidden {{.*}} <3 x i32> @test_load_offset_int3([[ENTRY_CXX]])
// SRV: %[[CALL:.*]] = call {{.*}} <3 x i32> @hlsl::[[TEXTURE]]<int vector[3]>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} @t_int3, [[LOAD_LLVM]] noundef %{{.*}}, [[OFFSET_LLVM]] noundef [[OFFSET_CONST]])
// SRV: ret <3 x i32> %[[CALL]]
int3 test_load_offset_int3(ENTRY_TYPE loc : LOC) {
  return t_int3.Load(LOAD_ARG, OFFSET_ARG);
}
#endif

// SRV: define linkonce_odr hidden {{.*}} <3 x i32> @hlsl::[[TEXTURE]]<int vector[3]>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]], [[OFFSET_LLVM]] {{.*}} %[[OFFSET:.*]])
// DXIL-SRV: %[[RES:.*]] = call <3 x i32> @llvm.dx.resource.load.level.v3i32.tdx.Texture_v3i32_{{.*}}("dx.Texture", <3 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SPIRV-SRV: %[[RES:.*]] = call <3 x i32> @llvm.spv.resource.load.level.v3i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], 0) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SRV: ret <3 x i32> %[[RES]]

TEXTURE<int4> t_int4;

// CHECK: define hidden {{.*}} <4 x i32> @test_load_int4([[ENTRY_CXX]])
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x i32> @hlsl::[[TEXTURE]]<int vector[4]>::Load([[LOAD_CXX]])(ptr {{.*}} @t_int4, [[LOAD_LLVM]] noundef %{{.*}})
// CHECK: ret <4 x i32> %[[CALL]]
int4 test_load_int4(ENTRY_TYPE loc : LOC) {
  return t_int4.Load(LOAD_ARG);
}

// CHECK: define linkonce_odr hidden {{.*}} <4 x i32> @hlsl::[[TEXTURE]]<int vector[4]>::Load([[LOAD_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]])
// DXIL: %[[RES:.*]] = call <4 x i32> @llvm.dx.resource.load.level.v4i32.tdx.Texture_v4i32_{{.*}}("dx.Texture", <4 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// SPIRV: %[[RES:.*]] = call <4 x i32> @llvm.spv.resource.load.level.v4i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT21]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 {{[^,]*}}, [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// CHECK: ret <4 x i32> %[[RES]]

#ifdef HAS_OFFSET
// SRV: define hidden {{.*}} <4 x i32> @test_load_offset_int4([[ENTRY_CXX]])
// SRV: %[[CALL:.*]] = call {{.*}} <4 x i32> @hlsl::[[TEXTURE]]<int vector[4]>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} @t_int4, [[LOAD_LLVM]] noundef %{{.*}}, [[OFFSET_LLVM]] noundef [[OFFSET_CONST]])
// SRV: ret <4 x i32> %[[CALL]]
int4 test_load_offset_int4(ENTRY_TYPE loc : LOC) {
  return t_int4.Load(LOAD_ARG, OFFSET_ARG);
}
#endif

// SRV: define linkonce_odr hidden {{.*}} <4 x i32> @hlsl::[[TEXTURE]]<int vector[4]>::Load([[LOAD_CXX]], [[OFFSET_CXX]])(ptr {{.*}} %[[THIS:.*]], [[LOAD_LLVM]] {{.*}} %[[LOAD:.*]], [[OFFSET_LLVM]] {{.*}} %[[OFFSET:.*]])
// DXIL-SRV: %[[RES:.*]] = call <4 x i32> @llvm.dx.resource.load.level.v4i32.tdx.Texture_v4i32_{{.*}}("dx.Texture", <4 x i32>, [[RW]], 0, 1, [[DXIL_TY]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SPIRV-SRV: %[[RES:.*]] = call <4 x i32> @llvm.spv.resource.load.level.v4i32.tspirv.SignedImage_i32_{{.*}}("spirv.SignedImage", i32, [[SPV_DIM]], 2, [[ARRAYED]], 0, [[SAMPLED]], [[FORMAT21]]) %{{.*}}, [[COORD_LLVM]] %{{.*}}, i32 %{{.*}}, [[OFFSET_LLVM]] %{{.*}})
// SRV: ret <4 x i32> %[[RES]]
