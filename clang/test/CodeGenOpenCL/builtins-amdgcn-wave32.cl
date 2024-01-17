// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -D__AMDGCN_WAVEFRONT_SIZE=32 -target-feature +wavefrontsize32 -S -emit-llvm -o - %s | FileCheck -enable-var-scope %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx1010 -S -emit-llvm -o - %s | FileCheck -enable-var-scope %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx1010 -target-feature +wavefrontsize32 -S -emit-llvm -o - %s | FileCheck -enable-var-scope %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -target-feature +wavefrontsize32 -S -emit-llvm -o - %s | FileCheck -enable-var-scope %s

typedef unsigned int uint;


// CHECK-LABEL: @test_ballot_wave32(
// CHECK: call i32 @llvm.amdgcn.ballot.i32(i1 %{{.+}})
void test_ballot_wave32(global uint* out, int a, int b)
{
  *out = __builtin_amdgcn_ballot_w32(a == b);
}

// CHECK: declare i32 @llvm.amdgcn.ballot.i32(i1) #[[$NOUNWIND_READONLY:[0-9]+]]

// CHECK-LABEL: @test_ballot_wave32_target_attr(
// CHECK: call i32 @llvm.amdgcn.ballot.i32(i1 %{{.+}})
__attribute__((target("wavefrontsize32")))
void test_ballot_wave32_target_attr(global uint* out, int a, int b)
{
  *out = __builtin_amdgcn_ballot_w32(a == b);
}

// CHECK-LABEL: @test_read_exec(
// CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 true)
void test_read_exec(global uint* out) {
  *out = __builtin_amdgcn_read_exec();
}

// CHECK: declare i64 @llvm.amdgcn.ballot.i64(i1) #[[$NOUNWIND_READONLY:[0-9]+]]

// CHECK-LABEL: @test_read_exec_lo(
// CHECK: call i32 @llvm.amdgcn.ballot.i32(i1 true)
void test_read_exec_lo(global uint* out) {
  *out = __builtin_amdgcn_read_exec_lo();
}

// CHECK-LABEL: @test_read_exec_hi(
// CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 true)
// CHECK: lshr i64 [[A:%.*]], 32
// CHECK: trunc i64 [[B:%.*]] to i32
void test_read_exec_hi(global uint* out) {
  *out = __builtin_amdgcn_read_exec_hi();
}

#if __AMDGCN_WAVEFRONT_SIZE != 32
#error Wrong wavesize detected
#endif
