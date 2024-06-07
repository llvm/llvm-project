// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1010 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1011 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1012 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm -o - %s | FileCheck %s

typedef unsigned int uint;
typedef unsigned long ulong;

// CHECK-LABEL: @test_permlane16(
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.permlane16(i32 %a, i32 %b, i32 %c, i32 %d, i1 false, i1 false)
void test_permlane16(global uint* out, uint a, uint b, uint c, uint d) {
  *out = __builtin_amdgcn_permlane16(a, b, c, d, 0, 0);
}

// CHECK-LABEL: @test_permlanex16(
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.permlanex16(i32 %a, i32 %b, i32 %c, i32 %d, i1 false, i1 false)
void test_permlanex16(global uint* out, uint a, uint b, uint c, uint d) {
  *out = __builtin_amdgcn_permlanex16(a, b, c, d, 0, 0);
}

// CHECK-LABEL: @test_mov_dpp8(
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.mov.dpp8.i32(i32 %a, i32 1)
void test_mov_dpp8(global uint* out, uint a) {
  *out = __builtin_amdgcn_mov_dpp8(a, 1);
}

// CHECK-LABEL: @test_s_memtime
// CHECK: {{.*}}call{{.*}} i64 @llvm.amdgcn.s.memtime()
void test_s_memtime(global ulong* out)
{
  *out = __builtin_amdgcn_s_memtime();
}

// CHECK-LABEL: @test_groupstaticsize
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.groupstaticsize()
void test_groupstaticsize(global uint* out)
{
  *out = __builtin_amdgcn_groupstaticsize();
}

// CHECK-LABEL: @test_ballot_wave32(
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.ballot.i32(i1 %{{.+}})
void test_ballot_wave32(global uint* out, int a, int b)
{
  *out = __builtin_amdgcn_ballot_w32(a == b);
}
