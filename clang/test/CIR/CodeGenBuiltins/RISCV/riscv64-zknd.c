// RUN: %clang_cc1 -triple riscv64 -target-feature +zknd -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR
// RUN: %clang_cc1 -triple riscv64 -target-feature +zknd -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +zknd -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM

// CIR-LABEL: cir.func{{.*}} @test_builtin_aes64dsm(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.aes64dsm" {{%.*}}, {{%.*}} : (!u64i, !u64i) -> !u64i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_aes64dsm(
// LLVM: call i64 @llvm.riscv.aes64dsm(i64 {{%.*}}, i64 {{%.*}})
// LLVM: ret i64
unsigned long test_builtin_aes64dsm(unsigned long a, unsigned long b) {
  return __builtin_riscv_aes64dsm(a, b);
}

// CIR-LABEL: cir.func{{.*}} @test_builtin_aes64ds(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.aes64ds" {{%.*}}, {{%.*}} : (!u64i, !u64i) -> !u64i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_aes64ds(
// LLVM: call i64 @llvm.riscv.aes64ds(i64 {{%.*}}, i64 {{%.*}})
// LLVM: ret i64
unsigned long test_builtin_aes64ds(unsigned long a, unsigned long b) {
  return __builtin_riscv_aes64ds(a, b);
}

// CIR-LABEL: cir.func{{.*}} @test_builtin_aes64im(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.aes64im" {{%.*}} : (!u64i) -> !u64i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_aes64im(
// LLVM: call i64 @llvm.riscv.aes64im(i64 {{%.*}})
// LLVM: ret i64
unsigned long test_builtin_aes64im(unsigned long a) {
  return __builtin_riscv_aes64im(a);
}
