// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -std=hlsl202x -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | FileCheck %s

// CHECK: %"class.hlsl::Texture2DArray" = type { target("dx.Texture", <4 x float>, 0, 0, 0, 7), %"struct.hlsl::Texture2DArray<>::mips_type" }
// CHECK: %"class.hlsl::Texture2DArray.0" = type { target("dx.Texture", float, 0, 0, 0, 7), %"struct.hlsl::Texture2DArray<float>::mips_type" }

// CHECK: @{{.*}}t1 = internal global %"class.hlsl::Texture2DArray" poison, align 4
Texture2DArray<> t1;

// CHECK: @{{.*}}t2 = internal global %"class.hlsl::Texture2DArray.0" poison, align 4
Texture2DArray<float> t2;

// CHECK: @{{.*}}t3 = internal global %"class.hlsl::Texture2DArray" poison, align 4
Texture2DArray t3;

void main() {
}
