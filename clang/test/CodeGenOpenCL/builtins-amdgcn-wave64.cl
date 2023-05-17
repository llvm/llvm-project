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

// CHECK-LABEL: @test_ballot_wave64_target_attr(
// CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 %{{.+}})
__attribute__((target("wavefrontsize64")))
void test_ballot_wave64_target_attr(global ulong* out, int a, int b)
{
  *out = __builtin_amdgcn_ballot_w64(a == b);
}

#if __AMDGCN_WAVEFRONT_SIZE != 64
#error Wrong wavesize detected
#endif
