// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu tonga -S -emit-llvm -o - %s | FileCheck %s

typedef unsigned long ulong;


// CHECK-LABEL: @test_s_memrealtime
// CHECK: call i64 @llvm.amdgcn.s.memrealtime()
void test_s_memrealtime(global ulong* out)
{
  *out = __builtin_amdgcn_s_memrealtime();
}
