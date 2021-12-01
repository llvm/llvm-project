// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s
// CHECK-LABEL: @test_non_temporal_store_kernel
// CHECK: store i32 0, i32 addrspace(1)* %{{.*}}, align 4, !tbaa !{{.*}}, !nontemporal {{.*}}

kernel void test_non_temporal_store_kernel(global unsigned int* io) {
  __builtin_nontemporal_store(0, io);
}
