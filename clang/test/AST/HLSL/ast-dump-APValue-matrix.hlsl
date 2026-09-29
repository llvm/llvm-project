// Test without serialization:
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x \
// RUN:            -fnative-half-type -fnative-int16-type \
// RUN:            -ast-dump %s -ast-dump-filter Test \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x \
// RUN:            -fnative-half-type -fnative-int16-type -emit-pch -o %t %s
// RUN: %clang_cc1 -x hlsl -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x \
// RUN:           -fnative-half-type -fnative-int16-type \
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

  constexpr int16_t3x2 i16mat3x2 = {-1, 2, -3, 4, -5, 6};
  // CHECK: VarDecl {{.*}} i16mat3x2 {{.*}} constexpr cinit
  // CHECK-NEXT: |-value: Matrix 3x2
  // CHECK-NEXT: | |-elements: Int -1, Int 2, Int -3, Int 4
  // CHECK-NEXT: | `-elements: Int -5, Int 6

  constexpr int64_t4x1 i64mat4x1 = {100, -200, 300, -400};
  // CHECK: VarDecl {{.*}} i64mat4x1 {{.*}} constexpr cinit
  // CHECK-NEXT: |-value: Matrix 4x1
  // CHECK-NEXT: | `-elements: Int 100, Int -200, Int 300, Int -400

  constexpr half2x3 hmat2x3 = {1.5h, -2.5h, 3.5h, -4.5h, 5.5h, -6.5h};
  // CHECK: VarDecl {{.*}} hmat2x3 {{.*}} constexpr cinit
  // CHECK-NEXT: |-value: Matrix 2x3
  // CHECK-NEXT: | |-elements: Float 1.500000e+00, Float -2.500000e+00, Float 3.500000e+00, Float -4.500000e+00
  // CHECK-NEXT: | `-elements: Float 5.500000e+00, Float -6.500000e+00

  constexpr double1x4 dmat1x4 = {0.5l, -1.25l, 2.75l, -3.125l};
  // CHECK: VarDecl {{.*}} dmat1x4 {{.*}} constexpr cinit
  // CHECK-NEXT: |-value: Matrix 1x4
  // CHECK-NEXT: | `-elements: Float 5.000000e-01, Float -1.250000e+00, Float 2.750000e+00, Float -3.125000e+00

  constexpr bool3x3 bmat3x3 = {true, false, true, false, true, false, true, false, true};
  // CHECK: VarDecl {{.*}} bmat3x3 {{.*}} constexpr cinit
  // CHECK-NEXT: |-value: Matrix 3x3
  // CHECK-NEXT: | |-elements: Int 1, Int 0, Int 1, Int 0
  // CHECK-NEXT: | |-elements: Int 1, Int 0, Int 1, Int 0
  // CHECK-NEXT: | `-element: Int 1
}
