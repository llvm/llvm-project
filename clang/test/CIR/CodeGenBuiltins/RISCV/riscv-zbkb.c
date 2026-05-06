// RUN: %clang_cc1 -triple riscv32 -target-feature +zbkb -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR32
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbkb -fclangir -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR64
// RUN: %clang_cc1 -triple riscv32 -target-feature +zbkb -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM32
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbkb -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM64
// RUN: %clang_cc1 -triple riscv32 -target-feature +zbkb -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM32
// RUN: %clang_cc1 -triple riscv64 -target-feature +zbkb -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM64

// CIR-LABEL: cir.func{{.*}} @test_builtin_brev8_32(
// CIR: {{%.*}} = cir.call_llvm_intrinsic "riscv.brev8" {{%.*}} : (!u32i) -> !u32i
// CIR: cir.return
// LLVM-LABEL: @test_builtin_brev8_32(
// LLVM: call i32 @llvm.riscv.brev8.i32(i32 {{%.*}})
// LLVM: ret i32
unsigned int test_builtin_brev8_32(unsigned int a) {
  return __builtin_riscv_brev8_32(a);
}

#if __riscv_xlen == 32
// CIR32-LABEL: cir.func{{.*}} @test_builtin_zip_32(
// CIR32: {{%.*}} = cir.call_llvm_intrinsic "riscv.zip" {{%.*}} : (!u32i) -> !u32i
// CIR32: cir.return
// LLVM32-LABEL: @test_builtin_zip_32(
// LLVM32: call i32 @llvm.riscv.zip.i32(i32 {{%.*}})
// LLVM32: ret i32
unsigned int test_builtin_zip_32(unsigned int a) {
  return __builtin_riscv_zip_32(a);
}

// CIR32-LABEL: cir.func{{.*}} @test_builtin_unzip_32(
// CIR32: {{%.*}} = cir.call_llvm_intrinsic "riscv.unzip" {{%.*}} : (!u32i) -> !u32i
// CIR32: cir.return
// LLVM32-LABEL: @test_builtin_unzip_32(
// LLVM32: call i32 @llvm.riscv.unzip.i32(i32 {{%.*}})
// LLVM32: ret i32
unsigned int test_builtin_unzip_32(unsigned int a) {
  return __builtin_riscv_unzip_32(a);
}
#endif

#if __riscv_xlen == 64
// CIR64-LABEL: cir.func{{.*}} @test_builtin_brev8_64(
// CIR64: {{%.*}} = cir.call_llvm_intrinsic "riscv.brev8" {{%.*}} : (!u64i) -> !u64i
// CIR64: cir.return
// LLVM64-LABEL: @test_builtin_brev8_64(
// LLVM64: call i64 @llvm.riscv.brev8.i64(i64 {{%.*}})
// LLVM64: ret i64
unsigned long long test_builtin_brev8_64(unsigned long long a) {
  return __builtin_riscv_brev8_64(a);
}
#endif
