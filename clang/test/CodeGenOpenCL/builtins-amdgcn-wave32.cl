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

// CHECK-LABEL: @test_ballot_wave32_target_attr(
// CHECK: call i32 @llvm.amdgcn.ballot.i32(i1 %{{.+}})
__attribute__((target("wavefrontsize32")))
void test_ballot_wave32_target_attr(global uint* out, int a, int b)
{
  *out = __builtin_amdgcn_ballot_w32(a == b);
}

#if __AMDGCN_WAVEFRONT_SIZE != 32
#error Wrong wavesize detected
#endif
