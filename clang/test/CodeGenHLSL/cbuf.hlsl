// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: @a = external addrspace(2) externally_initialized global float, align 4
// CHECK: @b = external addrspace(2) externally_initialized global double, align 8
// CHECK: @c = external addrspace(2) externally_initialized global float, align 4
// CHECK: @d = external addrspace(2) externally_initialized global double, align 8

// CHECK: @[[CB:.+]] = external constant { float, double }
cbuffer A : register(b0, space2) {
  float a;
  double b;
}

// CHECK: @[[TB:.+]] = external constant { float, double }
tbuffer A : register(t2, space1) {
  float c;
  double d;
}

float foo() {
// CHECK: load float, ptr addrspace(2) @a, align 4
// CHECK: load double, ptr addrspace(2) @b, align 8
// CHECK: load float, ptr addrspace(2) @c, align 4
// CHECK: load double, ptr addrspace(2) @d, align 8
  return a + b + c*d;
}

// CHECK: !hlsl.cbufs = !{![[CBMD:[0-9]+]]}
// CHECK: ![[CBMD]] = !{ptr @[[CB]], i32 13, i32 0, i1 false, i32 0, i32 2}
