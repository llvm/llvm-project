// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -std=hlsl202x -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -std=hlsl202x -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - -fexperimental-new-constant-interpreter | FileCheck %s


/// This test converts V to a 1-element vector and then .xx to a 2-element vector.
// CHECK-LABEL: ToTwoInts
// CHECK: [[splat:%.*]] = insertelement <1 x i32> poison, i32 {{.*}}, i64 0
// CHECK: [[vec2:%.*]] = shufflevector <1 x i32> [[splat]], <1 x i32> poison, <2 x i32> zeroinitializer
// CHECK: ret <2 x i32> [[vec2]]
int2 ToTwoInts(int V){
  return V.xx;
}

export void fn() {
  // This compiling successfully verifies that the vector constant expression
  // gets truncated to an integer at compile time for instantiation.
  _Static_assert(((int)1.xxxx) + 0 == 1, "Woo!");

  // This compiling successfully verifies that the vector constant expression
  // gets truncated to a float at compile time for instantiation.
  _Static_assert(((float)1.0.xxxx) + 0.0 == 1.0, "Woo!");

  // This compiling successfully verifies that a vector can be truncated to a
  // smaller vector, then truncated to a float as a constant expression.
  _Static_assert(((float2)float4(6, 5, 4, 3)).x == 6, "Woo!");
}
