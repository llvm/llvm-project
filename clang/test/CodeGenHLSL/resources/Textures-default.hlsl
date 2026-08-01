// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -std=hlsl202x -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2D -o - %s | FileCheck %s -DTEXTURE=Texture2D -DDXIL_TY=2 -DRW=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -std=hlsl202x -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DArray -o - %s | FileCheck %s -DTEXTURE=Texture2DArray -DDXIL_TY=7 -DRW=0

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
