// RUN: %clang_cc1 -triple riscv32 -target-feature +xandesperf -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple riscv32 -target-feature +xandesperf -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple riscv32 -target-feature +xandesperf -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// CIR-LABEL: cir.func{{.*}} @test_ffb(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.nds.ffb" {{%.*}}, {{%.*}} : (!u32i, !u32i) -> !s32i
// CIR: cir.return
// LLVM-LABEL: @test_ffb(
// LLVM: call i32 @llvm.riscv.nds.ffb.i32(i32 {{%.*}}, i32 {{%.*}})
// LLVM: ret i32
int test_ffb(unsigned int a, unsigned int b) {
  return __builtin_riscv_nds_ffb_32(a, b);
}

// CIR-LABEL: cir.func{{.*}} @test_ffzmism(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.nds.ffzmism" {{%.*}}, {{%.*}} : (!u32i, !u32i) -> !s32i
// CIR: cir.return
// LLVM-LABEL: @test_ffzmism(
// LLVM: call i32 @llvm.riscv.nds.ffzmism.i32(i32 {{%.*}}, i32 {{%.*}})
// LLVM: ret i32
int test_ffzmism(unsigned int a, unsigned int b) {
  return __builtin_riscv_nds_ffzmism_32(a, b);
}

// CIR-LABEL: cir.func{{.*}} @test_ffmism(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.nds.ffmism" {{%.*}}, {{%.*}} : (!u32i, !u32i) -> !s32i
// CIR: cir.return
// LLVM-LABEL: @test_ffmism(
// LLVM: call i32 @llvm.riscv.nds.ffmism.i32(i32 {{%.*}}, i32 {{%.*}})
// LLVM: ret i32
int test_ffmism(unsigned int a, unsigned int b) {
  return __builtin_riscv_nds_ffmism_32(a, b);
}

// CIR-LABEL: cir.func{{.*}} @test_flmism(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.nds.flmism" {{%.*}}, {{%.*}} : (!u32i, !u32i) -> !s32i
// CIR: cir.return
// LLVM-LABEL: @test_flmism(
// LLVM: call i32 @llvm.riscv.nds.flmism.i32(i32 {{%.*}}, i32 {{%.*}})
// LLVM: ret i32
int test_flmism(unsigned int a, unsigned int b) {
  return __builtin_riscv_nds_flmism_32(a, b);
}
