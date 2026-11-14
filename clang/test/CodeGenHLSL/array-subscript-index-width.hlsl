// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple spirv-vulkan-library %s -emit-llvm -o - | FileCheck %s

// A 32-bit array subscript should index the array without widening to 64 bits.

// CHECK-LABEL: define {{.*}}float @{{.*}}subscript{{.*}}(i32
// CHECK: [[REM:%.*]] = urem i32 {{.*}}, 3
// CHECK-NOT: {{sext|zext}}
// CHECK: getelementptr {{.*}}[3 x float], ptr {{.*}}, i32 0, i32 [[REM]]
export float subscript(uint i) {
  float l[3] = {1, 2, 3};
  return l[i % 3];
}
