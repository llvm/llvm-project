// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1101 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1102 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1103 -S -emit-llvm -o - %s | FileCheck %s

typedef unsigned int uint;
typedef unsigned long ulong;

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
