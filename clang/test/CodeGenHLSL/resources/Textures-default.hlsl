// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=Texture2D -o - %s \
// RUN:   | FileCheck %s -DTEXTURE=Texture2D -DDXIL_TY=2 -DRW=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -std=hlsl202x -emit-llvm -disable-llvm-passes \
// RUN:   -finclude-default-header -DTEXTURE=Texture2DArray -o - %s \
// RUN:   | FileCheck %s -DTEXTURE=Texture2DArray -DDXIL_TY=7 -DRW=0

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   DXIL_TY            dx.Texture resource-kind operand
//   RW                 dx.Texture UAV operand

// CHECK: %"class.hlsl::[[TEXTURE]]" = type { target("dx.Texture", <4 x float>, [[RW]], 0, 0, [[DXIL_TY]]), %"struct.hlsl::[[TEXTURE]]<>::mips_type" }
// CHECK: %"class.hlsl::[[TEXTURE]].0" = type { target("dx.Texture", float, [[RW]], 0, 0, [[DXIL_TY]]), %"struct.hlsl::[[TEXTURE]]<float>::mips_type" }

// CHECK: @{{.*}}t1 = internal global %"class.hlsl::[[TEXTURE]]" poison, align 4
TEXTURE<> t1;

// CHECK: @{{.*}}t2 = internal global %"class.hlsl::[[TEXTURE]].0" poison, align 4
TEXTURE<float> t2;

// CHECK: @{{.*}}t3 = internal global %"class.hlsl::[[TEXTURE]]" poison, align 4
TEXTURE t3;

void main() {
}
