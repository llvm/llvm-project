// RUN: %clang_cc1 -triple riscv32 -target-feature +zbc -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR32
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbc -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR64
// RUN: %clang_cc1 -triple riscv32 -target-feature +zbc -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM32
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbc -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM64
// RUN: %clang_cc1 -triple riscv32 -target-feature +zbc -emit-llvm %s -o - | FileCheck %s --check-prefixes=OGCG,OGCG32
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbc -emit-llvm %s -o - | FileCheck %s --check-prefixes=OGCG,OGCG64

// CIR-LABEL: cir.func{{.*}} @test_builtin_clmul_32(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "clmul" {{%.*}}, {{%.*}} : (!u32i, !u32i) -> !u32i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_clmul_32(
// LLVM: call i32 @llvm.clmul.i32(i32 {{%.*}}, i32 {{%.*}})
// LLVM: ret i32
// OGCG-LABEL: @test_builtin_clmul_32(
// OGCG: call i32 @llvm.clmul.i32(i32 {{%.*}}, i32 {{%.*}})
// OGCG: ret i32
unsigned int test_builtin_clmul_32(unsigned int a, unsigned int b) {
  return __builtin_riscv_clmul_32(a, b);
}

#if __riscv_xlen == 32
// CIR32-LABEL: cir.func{{.*}} @test_builtin_clmulh_32(
// CIR32: {{%.*}} = cir.call_llvm_intrinsic "riscv.clmulh" {{%.*}}, {{%.*}} : (!u32i, !u32i) -> !u32i
// CIR32: cir.return
// LLVM32-LABEL: @test_builtin_clmulh_32(
// LLVM32: call i32 @llvm.riscv.clmulh.i32(i32 {{%.*}}, i32 {{%.*}})
// LLVM32: ret i32
// OGCG32-LABEL: @test_builtin_clmulh_32(
// OGCG32: call i32 @llvm.riscv.clmulh.i32(i32 {{%.*}}, i32 {{%.*}})
// OGCG32: ret i32
unsigned int test_builtin_clmulh_32(unsigned int a, unsigned int b) {
  return __builtin_riscv_clmulh_32(a, b);
}

// CIR32-LABEL: cir.func{{.*}} @test_builtin_clmulr_32(
// CIR32: {{%.*}} = cir.call_llvm_intrinsic "riscv.clmulr" {{%.*}}, {{%.*}} : (!u32i, !u32i) -> !u32i
// CIR32: cir.return
// LLVM32-LABEL: @test_builtin_clmulr_32(
// LLVM32: call i32 @llvm.riscv.clmulr.i32(i32 {{%.*}}, i32 {{%.*}})
// LLVM32: ret i32
// OGCG32-LABEL: @test_builtin_clmulr_32(
// OGCG32: call i32 @llvm.riscv.clmulr.i32(i32 {{%.*}}, i32 {{%.*}})
// OGCG32: ret i32
unsigned int test_builtin_clmulr_32(unsigned int a, unsigned int b) {
  return __builtin_riscv_clmulr_32(a, b);
}
#endif

#if __riscv_xlen == 64
// CIR64-LABEL: cir.func{{.*}} @test_builtin_clmul_64(
// CIR64: {{%.*}} = cir.call_llvm_intrinsic "clmul" {{%.*}}, {{%.*}} : (!u64i, !u64i) -> !u64i
// CIR64: cir.return
// LLVM64-LABEL: @test_builtin_clmul_64(
// LLVM64: call i64 @llvm.clmul.i64(i64 {{%.*}}, i64 {{%.*}})
// LLVM64: ret i64
// OGCG64-LABEL: @test_builtin_clmul_64(
// OGCG64: call i64 @llvm.clmul.i64(i64 {{%.*}}, i64 {{%.*}})
// OGCG64: ret i64
unsigned long long test_builtin_clmul_64(unsigned long long a,
                                         unsigned long long b) {
  return __builtin_riscv_clmul_64(a, b);
}

// CIR64-LABEL: cir.func{{.*}} @test_builtin_clmulh_64(
// CIR64: {{%.*}} = cir.call_llvm_intrinsic "riscv.clmulh" {{%.*}}, {{%.*}} : (!u64i, !u64i) -> !u64i
// CIR64: cir.return
// LLVM64-LABEL: @test_builtin_clmulh_64(
// LLVM64: call i64 @llvm.riscv.clmulh.i64(i64 {{%.*}}, i64 {{%.*}})
// LLVM64: ret i64
// OGCG64-LABEL: @test_builtin_clmulh_64(
// OGCG64: call i64 @llvm.riscv.clmulh.i64(i64 {{%.*}}, i64 {{%.*}})
// OGCG64: ret i64
unsigned long long test_builtin_clmulh_64(unsigned long long a,
                                          unsigned long long b) {
  return __builtin_riscv_clmulh_64(a, b);
}

// CIR64-LABEL: cir.func{{.*}} @test_builtin_clmulr_64(
// CIR64: {{%.*}} = cir.call_llvm_intrinsic "riscv.clmulr" {{%.*}}, {{%.*}} : (!u64i, !u64i) -> !u64i
// CIR64: cir.return
// LLVM64-LABEL: @test_builtin_clmulr_64(
// LLVM64: call i64 @llvm.riscv.clmulr.i64(i64 {{%.*}}, i64 {{%.*}})
// LLVM64: ret i64
// OGCG64-LABEL: @test_builtin_clmulr_64(
// OGCG64: call i64 @llvm.riscv.clmulr.i64(i64 {{%.*}}, i64 {{%.*}})
// OGCG64: ret i64
unsigned long long test_builtin_clmulr_64(unsigned long long a,
                                          unsigned long long b) {
  return __builtin_riscv_clmulr_64(a, b);
}
#endif
