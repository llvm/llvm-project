// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after-all %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -mmlir --mlir-print-ir-after-all -mllvm -print-after-all  %s -o %t.ll 2>&1 | FileCheck %s -check-prefix=CIR -check-prefix=LLVM

int foo(void) {
  int i = 3;
  return i;
}

// CIR:  IR Dump After CIRCanonicalize (cir-canonicalize)
// CIR:  cir.func{{.*}} @foo() -> !s32i
// LLVM: IR Dump After cir::direct::ConvertCIRToLLVMPass (cir-flat-to-llvm)
// LLVM: llvm.func @foo() -> i32
// LLVM: IR Dump After
// LLVM: define{{.*}} i32 @foo()
