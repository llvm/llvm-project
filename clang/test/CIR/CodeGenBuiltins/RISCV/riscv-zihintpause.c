// RUN: %clang_cc1 -triple riscv32 -target-feature +zihintpause -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple riscv64 -target-feature +zihintpause -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple riscv32 -target-feature +zihintpause -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +zihintpause -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple riscv32 -target-feature +zihintpause -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG
// RUN: %clang_cc1 -triple riscv64 -target-feature +zihintpause -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG

void test_builtin_pause(void) {
  __builtin_riscv_pause();
}

// CIR-LABEL: cir.func{{.*}} @test_builtin_pause(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.pause" : () -> !void
// CIR: cir.return

// LLVM-LABEL: define dso_local void @test_builtin_pause(
// LLVM: call void @llvm.riscv.pause()
// LLVM: ret void

// OGCG-LABEL: define dso_local void @test_builtin_pause(
// OGCG: call void @llvm.riscv.pause()
// OGCG: ret void
