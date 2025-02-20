// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Make sure cbuffer inside namespace works.

// CHECK: @_ZN2n02n11aE = external addrspace(2) externally_initialized global float, align 4
// CHECK: @_ZN2n01bE = external addrspace(2) externally_initialized global float, align 4

// CHECK: @[[CB:.+]] = external constant { float }
// CHECK: @[[TB:.+]] = external constant { float }
namespace n0 {
namespace n1 {
  cbuffer A {
    float a;
  }
}
  tbuffer B {
    float b;
  }
}

float foo() {
// CHECK: load float, ptr addrspace(2) @_ZN2n02n11aE, align 4
// CHECK: load float, ptr addrspace(2) @_ZN2n01bE, align 4
  return n0::n1::a + n0::b;
}
