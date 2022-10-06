// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1101 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1102 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1103 -S -emit-llvm -o - %s | FileCheck %s

typedef unsigned int uint;
typedef unsigned long ulong;
typedef uint uint2 __attribute__((ext_vector_type(2)));
typedef uint uint4 __attribute__((ext_vector_type(4)));

// CHECK-LABEL: @test_s_sendmsg_rtn(
// CHECK: call i32 @llvm.amdgcn.s.sendmsg.rtn.i32(i32 0)
void test_s_sendmsg_rtn(global uint* out) {
  *out = __builtin_amdgcn_s_sendmsg_rtn(0);
}

// CHECK-LABEL: @test_s_sendmsg_rtnl(
// CHECK: call i64 @llvm.amdgcn.s.sendmsg.rtn.i64(i32 0)
void test_s_sendmsg_rtnl(global ulong* out) {
  *out = __builtin_amdgcn_s_sendmsg_rtnl(0);
}

// CHECK-LABEL: @test_ds_bvh_stack_rtn(
// CHECK: %0 = tail call { i32, i32 } @llvm.amdgcn.ds.bvh.stack.rtn(i32 %addr, i32 %data, <4 x i32> %data1, i32 128)
// CHECK: %1 = extractvalue { i32, i32 } %0, 0
// CHECK: %2 = extractvalue { i32, i32 } %0, 1
// CHECK: %3 = insertelement <2 x i32> poison, i32 %1, i64 0
// CHECK: %4 = insertelement <2 x i32> %3, i32 %2, i64 1
void test_ds_bvh_stack_rtn(global uint2* out, uint addr, uint data, uint4 data1)
{
  *out = __builtin_amdgcn_ds_bvh_stack_rtn(addr, data, data1, 128);
}
