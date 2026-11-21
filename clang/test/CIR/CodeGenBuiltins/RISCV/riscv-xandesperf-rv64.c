// RUN: %clang_cc1 -triple riscv64 -target-feature +xandesperf -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple riscv64 -target-feature +xandesperf -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +xandesperf -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// CIR-LABEL: cir.func{{.*}} @test_ffb(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.nds.ffb" {{%.*}}, {{%.*}} : (!u64i, !u64i) -> !s64i
// CIR: cir.return
// LLVM-LABEL: @test_ffb(
// LLVM: call i64 @llvm.riscv.nds.ffb.i64(i64 {{%.*}}, i64 {{%.*}})
// LLVM: ret i64
long long test_ffb(unsigned long long a, unsigned long long b) {
  return __builtin_riscv_nds_ffb_64(a, b);
}

// CIR-LABEL: cir.func{{.*}} @test_ffzmism(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.nds.ffzmism" {{%.*}}, {{%.*}} : (!u64i, !u64i) -> !s64i
// CIR: cir.return
// LLVM-LABEL: @test_ffzmism(
// LLVM: call i64 @llvm.riscv.nds.ffzmism.i64(i64 {{%.*}}, i64 {{%.*}})
// LLVM: ret i64
long long test_ffzmism(unsigned long long a, unsigned long long b) {
  return __builtin_riscv_nds_ffzmism_64(a, b);
}

// CIR-LABEL: cir.func{{.*}} @test_ffmism(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.nds.ffmism" {{%.*}}, {{%.*}} : (!u64i, !u64i) -> !s64i
// CIR: cir.return
// LLVM-LABEL: @test_ffmism(
// LLVM: call i64 @llvm.riscv.nds.ffmism.i64(i64 {{%.*}}, i64 {{%.*}})
// LLVM: ret i64
long long test_ffmism(unsigned long long a, unsigned long long b) {
  return __builtin_riscv_nds_ffmism_64(a, b);
}

// CIR-LABEL: cir.func{{.*}} @test_flmism(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.nds.flmism" {{%.*}}, {{%.*}} : (!u64i, !u64i) -> !s64i
// CIR: cir.return
// LLVM-LABEL: @test_flmism(
// LLVM: call i64 @llvm.riscv.nds.flmism.i64(i64 {{%.*}}, i64 {{%.*}})
// LLVM: ret i64
long long test_flmism(unsigned long long a, unsigned long long b) {
  return __builtin_riscv_nds_flmism_64(a, b);
}
