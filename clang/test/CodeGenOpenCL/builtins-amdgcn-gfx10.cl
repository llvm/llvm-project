// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1010 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1011 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1012 -S -emit-llvm -o - %s | FileCheck %s

typedef unsigned int uint;

// CHECK-LABEL: @test_permlane16(
// CHECK: call i32 @llvm.amdgcn.permlane16(i32 %a, i32 %b, i32 %c, i32 %d, i1 false, i1 false)
void test_permlane16(global uint* out, uint a, uint b, uint c, uint d) {
  *out = __builtin_amdgcn_permlane16(a, b, c, d, 0, 0);
}

// CHECK-LABEL: @test_permlanex16(
// CHECK: call i32 @llvm.amdgcn.permlanex16(i32 %a, i32 %b, i32 %c, i32 %d, i1 false, i1 false)
void test_permlanex16(global uint* out, uint a, uint b, uint c, uint d) {
  *out = __builtin_amdgcn_permlanex16(a, b, c, d, 0, 0);
}

// CHECK-LABEL: @test_mov_dpp8(
// CHECK: call i32 @llvm.amdgcn.mov.dpp8.i32(i32 %a, i32 1)
void test_mov_dpp8(global uint* out, uint a) {
  *out = __builtin_amdgcn_mov_dpp8(a, 1);
}
