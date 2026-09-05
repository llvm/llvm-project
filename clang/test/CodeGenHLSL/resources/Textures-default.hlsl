// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=Texture1D -o - %s | FileCheck %s \
// RUN:   -DTEXTURE=Texture1D -DDXIL_TY=1 -DRW=0 \
// RUN:   --check-prefixes=CHECK,CHECK-TEXEL -DCOORD_LLVM=float \
// RUN:   -DCOORD_CXX=float -DINDEX_LLVM=i32 \
// RUN:   -DINDEX_CXX_U="unsigned int" -DINDEX_CXX_I=int \
// RUN:   -DGRAD_LLVM=float -DGRAD_CXX=float -DOFFSET_LLVM=i32 \
// RUN:   -DOFFSET_CXX=int -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=Texture1DArray -o - %s | \
// RUN:   FileCheck %s -DTEXTURE=Texture1DArray -DDXIL_TY=6 -DRW=0 \
// RUN:   --check-prefixes=CHECK,CHECK-TEXEL -DCOORD_LLVM="<2 x float>" \
// RUN:   -DCOORD_CXX="float vector[2]" -DINDEX_LLVM="<2 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[2]" \
// RUN:   -DINDEX_CXX_I="int vector[2]" -DGRAD_LLVM=float \
// RUN:   -DGRAD_CXX=float -DOFFSET_LLVM=i32 -DOFFSET_CXX=int \
// RUN:   -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=Texture2D -o - %s | FileCheck %s \
// RUN:   -DTEXTURE=Texture2D -DDXIL_TY=2 -DRW=0 \
// RUN:   --check-prefixes=CHECK,CHECK-TEXEL -DCOORD_LLVM="<2 x float>" \
// RUN:   -DCOORD_CXX="float vector[2]" -DINDEX_LLVM="<2 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[2]" \
// RUN:   -DINDEX_CXX_I="int vector[2]" -DGRAD_LLVM="<2 x float>" \
// RUN:   -DGRAD_CXX="float vector[2]" -DOFFSET_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[2]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=TextureCube -o - %s | FileCheck \
// RUN:   %s -DTEXTURE=TextureCube -DDXIL_TY=5 -DRW=0 \
// RUN:   --check-prefixes=CHECK,CHECK-NOTEXEL -DCOORD_LLVM="<3 x float>" \
// RUN:   -DCOORD_CXX="float vector[3]" -DINDEX_LLVM="<3 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[3]" \
// RUN:   -DINDEX_CXX_I="int vector[3]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=TextureCubeArray -o - %s | \
// RUN:   FileCheck %s -DTEXTURE=TextureCubeArray -DDXIL_TY=9 -DRW=0 \
// RUN:   --check-prefixes=CHECK,CHECK-NOTEXEL -DCOORD_LLVM="<4 x float>" \
// RUN:   -DCOORD_CXX="float vector[4]" -DINDEX_LLVM="<4 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[4]" \
// RUN:   -DINDEX_CXX_I="int vector[4]" -DGRAD_LLVM="<3 x float>" \
// RUN:   -DGRAD_CXX="float vector[3]" -DOFFSET_LLVM="<3 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[3]" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=Texture2DArray -o - %s | \
// RUN:   FileCheck %s -DTEXTURE=Texture2DArray -DDXIL_TY=7 -DRW=0 \
// RUN:   --check-prefixes=CHECK,CHECK-TEXEL -DCOORD_LLVM="<3 x float>" \
// RUN:   -DCOORD_CXX="float vector[3]" -DINDEX_LLVM="<3 x i32>" \
// RUN:   -DINDEX_CXX_U="unsigned int vector[3]" \
// RUN:   -DINDEX_CXX_I="int vector[3]" -DGRAD_LLVM="<2 x float>" \
// RUN:   -DGRAD_CXX="float vector[2]" -DOFFSET_LLVM="<2 x i32>" \
// RUN:   -DOFFSET_CXX="int vector[2]" -DOFFSET_ZERO=zeroinitializer

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   GRAD_CXX           ddx/ddy type in the C++ signature
//   COORD_CXX          sample location type in the C++ signature
//   DXIL_TY            dx.Texture resource-kind operand
//   RW                 dx.Texture UAV operand
//
// Check prefixes:
//   TEXEL              the type has integer texel addressing (Load,
//                      operator[], mips), and therefore a `mips` field in its
//                      layout
//   NOTEXEL            the type has no integer texel addressing

// CHECK-TEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// CHECK-NOTEXEL: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]) }
// CHECK-TEXEL: %"class.hlsl::[[TEXTURE]].0" = type { target("dx.Texture", float, [[RW]], 0, 0, [[DXIL_TY]]), %"struct.hlsl::[[TEXTURE]]<float>::mips_type" }
// CHECK-NOTEXEL: %"class.hlsl::[[TEXTURE]].0" = type { target("dx.Texture", float, [[RW]], 0, 0, [[DXIL_TY]]) }

// CHECK: @{{.*}}t1 = internal global %"class.hlsl::[[TEXTURE]]" poison, align 4
TEXTURE<> t1;

// CHECK: @{{.*}}t2 = internal global %"class.hlsl::[[TEXTURE]].0" poison, align 4
TEXTURE<float> t2;

// CHECK: @{{.*}}t3 = internal global %"class.hlsl::[[TEXTURE]]" poison, align 4
TEXTURE t3;

void main() {
}
