// RUN: %clang_cc1 -triple riscv32 -target-feature +zksh -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR
// RUN: %clang_cc1 -triple riscv64 -target-feature +zksh -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR
// RUN: %clang_cc1 -triple riscv32 -target-feature +zksh -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +zksh -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -triple riscv32 -target-feature +zksh -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +zksh -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM

// CIR-LABEL: cir.func{{.*}} @test_builtin_sm3p0(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.sm3p0" {{%.*}} : (!u32i) -> !u32i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_sm3p0(
// LLVM: call i32 @llvm.riscv.sm3p0(i32 {{%.*}})
// LLVM: ret i32
unsigned int test_builtin_sm3p0(unsigned int a) {
  return __builtin_riscv_sm3p0(a);
}

// CIR-LABEL: cir.func{{.*}} @test_builtin_sm3p1(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.sm3p1" {{%.*}} : (!u32i) -> !u32i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_sm3p1(
// LLVM: call i32 @llvm.riscv.sm3p1(i32 {{%.*}})
// LLVM: ret i32
unsigned int test_builtin_sm3p1(unsigned int a) {
  return __builtin_riscv_sm3p1(a);
}
