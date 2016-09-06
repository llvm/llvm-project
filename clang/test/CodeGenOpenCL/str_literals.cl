// RUN: %clang_cc1 %s -cl-opt-disable -emit-llvm -o - -ffake-address-space-map | FileCheck %s

__constant char * __constant x = "hello world";
__constant char * __constant y = "hello world";

// CHECK: unnamed_addr addrspace(3) constant
// CHECK-NOT: addrspace(3) unnamed_addr constant
// CHECK: @x = addrspace(3) constant i8 addrspace(3)*
// CHECK: @y = addrspace(3) constant i8 addrspace(3)*
