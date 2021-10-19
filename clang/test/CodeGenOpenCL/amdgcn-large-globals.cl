// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s

// CHECK: @One ={{.*}} local_unnamed_addr addrspace(1) global [6442450944 x i8] zeroinitializer, align 1
unsigned char One[6442450944];
// CHECK: @Two ={{.*}} local_unnamed_addr addrspace(1) global [6442450944 x i32] zeroinitializer, align 4
global unsigned int Two[6442450944];

kernel void large_globals(unsigned int id) {
  One[id] = id;
  Two[id + 1] = id + 1;
}
