// RUN: %clang_cc1 -triple riscv32 -target-feature +zksed -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR
// RUN: %clang_cc1 -triple riscv64 -target-feature +zksed -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR
// RUN: %clang_cc1 -triple riscv32 -target-feature +zksed -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +zksed -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -triple riscv32 -target-feature +zksed -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +zksed -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM

// CIR-LABEL: cir.func{{.*}} @test_builtin_sm4ks(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.sm4ks" {{%.*}}, {{%.*}}, {{%.*}} : (!u32i, !u32i, !u32i) -> !u32i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_sm4ks(
// LLVM: call i32 @llvm.riscv.sm4ks(i32 {{%.*}}, i32 {{%.*}}, i32 0)
// LLVM: ret i32
unsigned int test_builtin_sm4ks(unsigned int a, unsigned int b) {
  return __builtin_riscv_sm4ks(a, b, 0);
}

// CIR-LABEL: cir.func{{.*}} @test_builtin_sm4ed(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.sm4ed" {{%.*}}, {{%.*}}, {{%.*}} : (!u32i, !u32i, !u32i) -> !u32i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_sm4ed(
// LLVM: call i32 @llvm.riscv.sm4ed(i32 {{%.*}}, i32 {{%.*}}, i32 0)
// LLVM: ret i32
unsigned int test_builtin_sm4ed(unsigned int a, unsigned int b) {
  return __builtin_riscv_sm4ed(a, b, 0);
}
