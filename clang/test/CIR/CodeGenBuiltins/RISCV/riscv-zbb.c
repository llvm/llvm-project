// RUN: %clang_cc1 -triple riscv32 -target-feature +zbb -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbb -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR64
// RUN: %clang_cc1 -triple riscv32 -target-feature +zbb -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbb -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM64
// RUN: %clang_cc1 -triple riscv32 -target-feature +zbb -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbb -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM64


// CIR-LABEL: cir.func{{.*}} @test_builtin_orc_b_32(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.orc.b" {{%.*}} : (!u32i) -> !u32i
// CIR: cir.return

// LLVM-LABEL: @test_builtin_orc_b_32(
// LLVM: call i32 @llvm.riscv.orc.b.i32(i32 {{%.*}})
// LLVM: ret i32
unsigned int test_builtin_orc_b_32(unsigned int a) {
  return __builtin_riscv_orc_b_32(a);
}

// CIR-LABEL: cir.func{{.*}} @test_builtin_clz_32(
// CIR: {{%.*}} = cir.clz {{%.*}} : !u32i
// CIR: cir.return

// LLVM-LABEL: @test_builtin_clz_32(
// LLVM: call i32 @llvm.ctlz.i32(i32 {{%.*}}, i1 false)
// LLVM: ret i32
unsigned int test_builtin_clz_32(unsigned int a) {
  return __builtin_riscv_clz_32(a);
}

// CIR-LABEL: cir.func{{.*}} @test_builtin_ctz_32(
// CIR: {{%.*}} = cir.ctz {{%.*}} : !u32i
// CIR: cir.return

// LLVM-LABEL: @test_builtin_ctz_32(
// LLVM: call i32 @llvm.cttz.i32(i32 {{%.*}}, i1 false)
// LLVM: ret i32
unsigned int test_builtin_ctz_32(unsigned int a) {
  return __builtin_riscv_ctz_32(a);
}

#if __riscv_xlen == 64
// CIR64-LABEL: cir.func{{.*}} @test_builtin_orc_b_64(
// CIR64: {{%.*}} = cir.call_llvm_intrinsic "riscv.orc.b" {{%.*}} : (!u64i) -> !u64i
// CIR64: cir.return

// LLVM64-LABEL: @test_builtin_orc_b_64(
// LLVM64: call i64 @llvm.riscv.orc.b.i64(i64 {{%.*}})
// LLVM64: ret i64

unsigned long long test_builtin_orc_b_64(unsigned long long a) {
  return __builtin_riscv_orc_b_64(a);
}

// CIR64-LABEL: cir.func{{.*}} @test_builtin_clz_64(
// CIR64: {{%.*}} = cir.clz {{%.*}} : !u64i
// CIR64: cir.return

// LLVM64-LABEL: @test_builtin_clz_64(
// LLVM64: call i64 @llvm.ctlz.i64(i64 {{%.*}}, i1 false)
// LLVM64: ret i64
unsigned long long test_builtin_clz_64(unsigned long long a) {
  return __builtin_riscv_clz_64(a);
}

// CIR64-LABEL: cir.func{{.*}} @test_builtin_ctz_64(
// CIR64: {{%.*}} = cir.ctz {{%.*}} : !u64i
// CIR64: cir.return

// LLVM64-LABEL: @test_builtin_ctz_64(
// LLVM64: call i64 @llvm.cttz.i64(i64 {{%.*}}, i1 false)
// LLVM64: ret i64
unsigned long long test_builtin_ctz_64(unsigned long long a) {
  return __builtin_riscv_ctz_64(a);
}
#endif
