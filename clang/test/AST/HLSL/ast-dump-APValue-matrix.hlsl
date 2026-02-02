// Test without serialization:
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x \
// RUN:            -ast-dump %s -ast-dump-filter Test \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x -emit-pch -o %t %s
// RUN: %clang_cc1 -x hlsl -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x \
// RUN:           -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

export void Test() {
  constexpr int2x2 mat2x2 = {1, 2, 3, 4};
  // CHECK: VarDecl {{.*}} mat2x2 {{.*}} constexpr cinit
  // CHECK-NEXT: |-value: Matrix 2x2
  // CHECK-NEXT: | `-elements: Int 1, Int 2, Int 3, Int 4

  constexpr float3x2 mat3x2 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  // CHECK: VarDecl {{.*}} mat3x2 {{.*}} constexpr cinit
  // CHECK-NEXT: |-value: Matrix 3x2
  // CHECK-NEXT: | |-elements: Float 1.000000e+00, Float 2.000000e+00, Float 3.000000e+00, Float 4.000000e+00
  // CHECK-NEXT: | `-elements: Float 5.000000e+00, Float 6.000000e+00
}
