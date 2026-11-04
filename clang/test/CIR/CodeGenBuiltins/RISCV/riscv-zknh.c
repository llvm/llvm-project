// RUN: %clang_cc1 -triple riscv32 -target-feature +zknh -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR
// RUN: %clang_cc1 -triple riscv64 -target-feature +zknh -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR
// RUN: %clang_cc1 -triple riscv32 -target-feature +zknh -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +zknh -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -triple riscv32 -target-feature +zknh -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +zknh -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM

// CIR-LABEL: cir.func{{.*}} @test_builtin_sha256sig0(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.sha256sig0" {{%.*}} : (!u32i) -> !u32i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_sha256sig0(
// LLVM: call i32 @llvm.riscv.sha256sig0(i32 {{%.*}})
// LLVM: ret i32
unsigned int test_builtin_sha256sig0(unsigned int a) {
  return __builtin_riscv_sha256sig0(a);
}

// CIR-LABEL: cir.func{{.*}} @test_builtin_sha256sig1(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.sha256sig1" {{%.*}} : (!u32i) -> !u32i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_sha256sig1(
// LLVM: call i32 @llvm.riscv.sha256sig1(i32 {{%.*}})
// LLVM: ret i32
unsigned int test_builtin_sha256sig1(unsigned int a) {
  return __builtin_riscv_sha256sig1(a);
}

// CIR-LABEL: cir.func{{.*}} @test_builtin_sha256sum0(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.sha256sum0" {{%.*}} : (!u32i) -> !u32i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_sha256sum0(
// LLVM: call i32 @llvm.riscv.sha256sum0(i32 {{%.*}})
// LLVM: ret i32
unsigned int test_builtin_sha256sum0(unsigned int a) {
  return __builtin_riscv_sha256sum0(a);
}

// CIR-LABEL: cir.func{{.*}} @test_builtin_sha256sum1(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.sha256sum1" {{%.*}} : (!u32i) -> !u32i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_sha256sum1(
// LLVM: call i32 @llvm.riscv.sha256sum1(i32 {{%.*}})
// LLVM: ret i32
unsigned int test_builtin_sha256sum1(unsigned int a) {
  return __builtin_riscv_sha256sum1(a);
}
