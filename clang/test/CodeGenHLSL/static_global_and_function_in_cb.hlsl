// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

cbuffer A {
  // CHECK: @a = external addrspace(2) externally_initialized global float, align 4
  float a;
  // CHECK: @_ZL1b = internal global float 3.000000e+00, align 4
  static float b = 3;
  float foo() { return a + b; }
}
// CHECK: @[[CB:.+]] = external constant { float }

// CHECK:define {{.*}} float @_Z3foov()
// CHECK:load float, ptr addrspace(2) @a, align 4
// CHECK:load float, ptr @_ZL1b, align 4

float bar() {
  return foo();
}
