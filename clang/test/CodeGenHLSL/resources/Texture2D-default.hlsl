// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -std=hlsl202x -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | FileCheck %s

// CHECK: %"class.hlsl::Texture2D" = type { target("dx.Texture", <4 x float>, 0, 0, 0, 2), %"struct.hlsl::Texture2D<>::mips_type" }
// CHECK: %"class.hlsl::Texture2D.0" = type { target("dx.Texture", float, 0, 0, 0, 2), %"struct.hlsl::Texture2D<float>::mips_type" }

// CHECK: @{{.*}}t1 = internal global %"class.hlsl::Texture2D" poison, align 4
Texture2D<> t1;

// CHECK: @{{.*}}t2 = internal global %"class.hlsl::Texture2D.0" poison, align 4
Texture2D<float> t2;

// CHECK: @{{.*}}t3 = internal global %"class.hlsl::Texture2D" poison, align 4
Texture2D t3;

void main() {
}
