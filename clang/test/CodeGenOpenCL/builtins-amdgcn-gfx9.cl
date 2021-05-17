// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx900 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1010 -S -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef unsigned int uint;
typedef unsigned long ulong;

// CHECK-LABEL: @test_fmed3_f16
// CHECK: call half @llvm.amdgcn.fmed3.f16(half %a, half %b, half %c)
void test_fmed3_f16(global half* out, half a, half b, half c)
{
  *out = __builtin_amdgcn_fmed3h(a, b, c);
}

// CHECK-LABEL: @test_s_memtime
// CHECK: call i64 @llvm.amdgcn.s.memtime()
void test_s_memtime(global ulong* out)
{
  *out = __builtin_amdgcn_s_memtime();
}

// CHECK-LABEL: @test_groupstaticsize
// CHECK: call i32 @llvm.amdgcn.groupstaticsize()
void test_groupstaticsize(global uint* out)
{
  *out = __builtin_amdgcn_groupstaticsize();
}
