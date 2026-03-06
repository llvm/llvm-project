// RUN: %clang_cc1 -triple x86_64-linux -target-cpu haswell -O2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -target-cpu haswell -O0 -emit-llvm -o - %s | FileCheck %s

// The intrinsics survive -emit-llvm (they are only resolved in the
// backend pipeline, not the mid-end). Verify they are emitted correctly.

void avx2_path(void);

void test_is_haswell(void) {
// CHECK-LABEL: @test_is_haswell
// CHECK: call i1 @llvm.target.is.cpu(metadata !"haswell")
  if (__builtin_target_is_cpu("haswell"))
    avx2_path();
}
