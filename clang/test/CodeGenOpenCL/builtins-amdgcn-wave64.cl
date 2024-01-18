// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-feature +wavefrontsize64 -S -emit-llvm -o - %s | FileCheck -enable-var-scope %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx900 -S -emit-llvm -o - %s | FileCheck -enable-var-scope %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx900 -target-feature +wavefrontsize64 -S -emit-llvm -o - %s | FileCheck -enable-var-scope %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx1010 -target-feature +wavefrontsize64 -S -emit-llvm -o - %s | FileCheck -enable-var-scope %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -target-feature +wavefrontsize64 -S -emit-llvm -o - %s | FileCheck -enable-var-scope %s

typedef unsigned long ulong;

// CHECK-LABEL: @test_ballot_wave64(
// CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 %{{.+}})
void test_ballot_wave64(global ulong* out, int a, int b)
{
  *out = __builtin_amdgcn_ballot_w64(a == b);
}

// CHECK: declare i64 @llvm.amdgcn.ballot.i64(i1) #[[$NOUNWIND_READONLY:[0-9]+]]

// CHECK-LABEL: @test_ballot_wave64_target_attr(
// CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 %{{.+}})
__attribute__((target("wavefrontsize64")))
void test_ballot_wave64_target_attr(global ulong* out, int a, int b)
{
  *out = __builtin_amdgcn_ballot_w64(a == b);
}

// CHECK-LABEL: @test_read_exec(
// CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 true)
void test_read_exec(global ulong* out) {
  *out = __builtin_amdgcn_read_exec();
}

// CHECK-LABEL: @test_read_exec_lo(
// CHECK: call i32 @llvm.amdgcn.ballot.i32(i1 true)
void test_read_exec_lo(global ulong* out) {
  *out = __builtin_amdgcn_read_exec_lo();
}

// CHECK: declare i32 @llvm.amdgcn.ballot.i32(i1) #[[$NOUNWIND_READONLY:[0-9]+]]

// CHECK-LABEL: @test_read_exec_hi(
// CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 true)
// CHECK: lshr i64 [[A:%.*]], 32
void test_read_exec_hi(global ulong* out) {
  *out = __builtin_amdgcn_read_exec_hi();
}

#if __AMDGCN_WAVEFRONT_SIZE != 64
#error Wrong wavesize detected
#endif
