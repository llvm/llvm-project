// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu hawaii -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu fiji -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx906 -S -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: @test_s_dcache_inv_vol
// CHECK: call void @llvm.amdgcn.s.dcache.inv.vol(
void test_s_dcache_inv_vol()
{
  __builtin_amdgcn_s_dcache_inv_vol();
}

// CHECK-LABEL: @test_buffer_wbinvl1_vol
// CHECK: call void @llvm.amdgcn.buffer.wbinvl1.vol()
void test_buffer_wbinvl1_vol()
{
  __builtin_amdgcn_buffer_wbinvl1_vol();
}

