// RUN: %clang_cc1 -triple riscv32 -target-feature +zknd -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR
// RUN: %clang_cc1 -triple riscv32 -target-feature +zknd -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -triple riscv32 -target-feature +zknd -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM

// CIR-LABEL: cir.func{{.*}} @test_builtin_aes32dsi(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.aes32dsi" {{%.*}}, {{%.*}}, {{%.*}} : (!u32i, !u32i, !u32i) -> !u32i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_aes32dsi(
// LLVM: call i32 @llvm.riscv.aes32dsi(i32 {{%.*}}, i32 {{%.*}}, i32 3)
// LLVM: ret i32
unsigned int test_builtin_aes32dsi(unsigned int a, unsigned int b) {
  return __builtin_riscv_aes32dsi(a, b, 3);
}

// CIR-LABEL: cir.func{{.*}} @test_builtin_aes32dsmi(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.aes32dsmi" {{%.*}}, {{%.*}}, {{%.*}} : (!u32i, !u32i, !u32i) -> !u32i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_aes32dsmi(
// LLVM: call i32 @llvm.riscv.aes32dsmi(i32 {{%.*}}, i32 {{%.*}}, i32 3)
// LLVM: ret i32
unsigned int test_builtin_aes32dsmi(unsigned int a, unsigned int b) {
  return __builtin_riscv_aes32dsmi(a, b, 3);
}
