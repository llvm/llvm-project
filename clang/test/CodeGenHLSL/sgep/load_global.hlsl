// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -fexperimental-emit-sgep -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -fexperimental-emit-sgep -o - %s | FileCheck %s --check-prefixes=CHECK-SPIR

struct S {
  uint a;
  uint b;
  uint c;
  uint d;
};

// CHECK-DXIL: @_ZL1s = external hidden addrspace(2) global %struct.S, align 1
// CHECK-DXIL: @_ZL1a = external hidden addrspace(2) constant [4 x i32], align 4
// CHECK-DXIL: @_ZL1b = external hidden addrspace(2) constant i32, align 4

// CHECK-SPIR: @_ZL1s = external hidden addrspace(12) global %struct.S, align 1
// CHECK-SPIR: @_ZL1a = external hidden addrspace(12) constant [4 x i32], align 4
// CHECK-SPIR: @_ZL1b = external hidden addrspace(12) constant i32, align 4
const S s;
const uint a[4];
const uint b;

void foo() {

// CHECK-DXIL: %[[#PTR:]] = call ptr addrspace(2) (ptr addrspace(2), ...) @llvm.structured.gep.p2(ptr addrspace(2) elementtype(%S) @_ZL1s, i32 1)
// CHECK-DXIL: %[[#]] = load i32, ptr addrspace(2) %[[#PTR]], align 4

// CHECK-SPIR: %[[#PTR:]] = call ptr addrspace(12) (ptr addrspace(12), ...) @llvm.structured.gep.p12(ptr addrspace(12) elementtype(%S) @_ZL1s, i32 1)
// CHECK-SPIR: %[[#]] = load i32, ptr addrspace(12) %[[#PTR]], align 4
  uint tmp = s.b;
}

void bar() {
// CHECK-DXIL: %cbufferidx = call ptr addrspace(2) (ptr addrspace(2), ...) @llvm.structured.gep.p2(ptr addrspace(2) elementtype([4 x <{ i32, target("dx.Padding", 12) }>]) @_ZL1a, i32 2, i32 0)
// CHECK-DXIL: %[[#]] = load i32, ptr addrspace(2) %cbufferidx, align 16

// CHECK-SPIR: %cbufferidx = call ptr addrspace(12) (ptr addrspace(12), ...) @llvm.structured.gep.p12(ptr addrspace(12) elementtype([4 x <{ i32, target("spirv.Padding", 12) }>]) @_ZL1a, i64 2, i32 0)
// CHECK-SPIR: %[[#]] = load i32, ptr addrspace(12) %cbufferidx, align 16
  uint tmp = a[2];
}

void baz() {
// CHECK-DXIL: %[[#]] = load i32, ptr addrspace(2) @_ZL1b, align 4
// CHECK-SPIR: %[[#]] = load i32, ptr addrspace(12) @_ZL1b, align 4
  uint tmp = b;
}
