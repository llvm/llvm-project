// RUN: %clang_cc1 -triple riscv32 -target-feature +zbb -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbb -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR64
// RUN: %clang_cc1 -triple riscv32 -target-feature +zbb -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbb -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM64
// RUN: %clang_cc1 -triple riscv32 -target-feature +zbb -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbb -emit-llvm %s -o - | FileCheck %s --check-prefixes=OGCG,OGCG64

unsigned int test_builtin_orc_b_32(unsigned int a) {
  return __builtin_riscv_orc_b_32(a);
}

#if __riscv_xlen == 64
unsigned long long test_builtin_orc_b_64(unsigned long long a) {
  return __builtin_riscv_orc_b_64(a);
}
#endif

// CIR-LABEL: cir.func{{.*}} @test_builtin_orc_b_32(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.orc.b" {{%.*}} : (!u32i) -> !u32i
// CIR: cir.return

// CIR64-LABEL: cir.func{{.*}} @test_builtin_orc_b_64(
// CIR64: {{%.*}} = cir.call_llvm_intrinsic "riscv.orc.b" {{%.*}} : (!u64i) -> !u64i
// CIR64: cir.return

// LLVM-LABEL: @test_builtin_orc_b_32(
// LLVM: call i32 @llvm.riscv.orc.b.i32(i32 {{%.*}})
// LLVM: ret i32

// LLVM64-LABEL: @test_builtin_orc_b_64(
// LLVM64: call i64 @llvm.riscv.orc.b.i64(i64 {{%.*}})
// LLVM64: ret i64

// OGCG-LABEL: @test_builtin_orc_b_32(
// OGCG: call i32 @llvm.riscv.orc.b.i32(i32 {{%.*}})
// OGCG: ret i32

// OGCG64-LABEL: @test_builtin_orc_b_64(
// OGCG64: call i64 @llvm.riscv.orc.b.i64(i64 {{%.*}})
// OGCG64: ret i64
