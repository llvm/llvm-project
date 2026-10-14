// RUN: %clang_cc1 -triple riscv32 -target-feature +zbkx -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR32
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbkx -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR64
// RUN: %clang_cc1 -triple riscv32 -target-feature +zbkx -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM32
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbkx -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM64
// RUN: %clang_cc1 -triple riscv32 -target-feature +zbkx -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG32
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbkx -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG64

#if __riscv_xlen == 32
// CIR32-LABEL: cir.func{{.*}} @test_builtin_xperm4_32(
// CIR32: {{%.*}} = cir.call_llvm_intrinsic "riscv.xperm4" {{%.*}}, {{%.*}} : (!u32i, !u32i) -> !u32i
// CIR32: cir.return
// LLVM32-LABEL: @test_builtin_xperm4_32(
// LLVM32: call i32 @llvm.riscv.xperm4.i32(i32 {{%.*}}, i32 {{%.*}})
// LLVM32: ret i32
// OGCG32-LABEL: @test_builtin_xperm4_32(
// OGCG32: call i32 @llvm.riscv.xperm4.i32(i32 {{%.*}}, i32 {{%.*}})
// OGCG32: ret i32
unsigned int test_builtin_xperm4_32(unsigned int a, unsigned int b) {
  return __builtin_riscv_xperm4_32(a, b);
}

// CIR32-LABEL: cir.func{{.*}} @test_builtin_xperm8_32(
// CIR32: {{%.*}} = cir.call_llvm_intrinsic "riscv.xperm8" {{%.*}}, {{%.*}} : (!u32i, !u32i) -> !u32i
// CIR32: cir.return
// LLVM32-LABEL: @test_builtin_xperm8_32(
// LLVM32: call i32 @llvm.riscv.xperm8.i32(i32 {{%.*}}, i32 {{%.*}})
// LLVM32: ret i32
// OGCG32-LABEL: @test_builtin_xperm8_32(
// OGCG32: call i32 @llvm.riscv.xperm8.i32(i32 {{%.*}}, i32 {{%.*}})
// OGCG32: ret i32
unsigned int test_builtin_xperm8_32(unsigned int a, unsigned int b) {
  return __builtin_riscv_xperm8_32(a, b);
}
#endif

#if __riscv_xlen == 64
// CIR64-LABEL: cir.func{{.*}} @test_builtin_xperm4_64(
// CIR64: {{%.*}} = cir.call_llvm_intrinsic "riscv.xperm4" {{%.*}}, {{%.*}} : (!u64i, !u64i) -> !u64i
// CIR64: cir.return
// LLVM64-LABEL: @test_builtin_xperm4_64(
// LLVM64: call i64 @llvm.riscv.xperm4.i64(i64 {{%.*}}, i64 {{%.*}})
// LLVM64: ret i64
// OGCG64-LABEL: @test_builtin_xperm4_64(
// OGCG64: call i64 @llvm.riscv.xperm4.i64(i64 {{%.*}}, i64 {{%.*}})
// OGCG64: ret i64
unsigned long long test_builtin_xperm4_64(unsigned long long a,
                                          unsigned long long b) {
  return __builtin_riscv_xperm4_64(a, b);
}

// CIR64-LABEL: cir.func{{.*}} @test_builtin_xperm8_64(
// CIR64: {{%.*}} = cir.call_llvm_intrinsic "riscv.xperm8" {{%.*}}, {{%.*}} : (!u64i, !u64i) -> !u64i
// CIR64: cir.return
// LLVM64-LABEL: @test_builtin_xperm8_64(
// LLVM64: call i64 @llvm.riscv.xperm8.i64(i64 {{%.*}}, i64 {{%.*}})
// LLVM64: ret i64
// OGCG64-LABEL: @test_builtin_xperm8_64(
// OGCG64: call i64 @llvm.riscv.xperm8.i64(i64 {{%.*}}, i64 {{%.*}})
// OGCG64: ret i64
unsigned long long test_builtin_xperm8_64(unsigned long long a,
                                          unsigned long long b) {
  return __builtin_riscv_xperm8_64(a, b);
}
#endif
