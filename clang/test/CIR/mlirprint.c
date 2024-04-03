// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after-all %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir-flat -mmlir --mlir-print-ir-after-all %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIRFLAT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -mmlir --mlir-print-ir-after-all -mllvm -print-after-all  %s -o %t.ll 2>&1 | FileCheck %s -check-prefix=CIR -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-drop-ast %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIRPASS
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir-flat -mmlir --mlir-print-ir-before=cir-flatten-cfg %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CFGPASS

int foo(void) {
  int i = 3;
  return i;
}


// CIR:  IR Dump After MergeCleanups (cir-merge-cleanups)
// CIR:  cir.func @foo() -> !s32i
// CIR:  IR Dump After LoweringPrepare (cir-lowering-prepare)
// CIR:  cir.func @foo() -> !s32i
// CIR-NOT: IR Dump After FlattenCFG
// CIR:  IR Dump After DropAST (cir-drop-ast)
// CIR:  cir.func @foo() -> !s32i
// CIRFLAT:  IR Dump After MergeCleanups (cir-merge-cleanups)
// CIRFLAT:  cir.func @foo() -> !s32i
// CIRFLAT:  IR Dump After LoweringPrepare (cir-lowering-prepare)
// CIRFLAT:  cir.func @foo() -> !s32i
// CIRFLAT:  IR Dump After FlattenCFG (cir-flatten-cfg)
// CIRFLAT:  IR Dump After DropAST (cir-drop-ast)
// CIRFLAT:  cir.func @foo() -> !s32i
// LLVM: IR Dump After cir::direct::ConvertCIRToLLVMPass (cir-to-llvm-internal)
// LLVM: llvm.func @foo() -> i32
// LLVM: IR Dump After
// LLVM: define i32 @foo()

// CIRPASS-NOT:  IR Dump After MergeCleanups
// CIRPASS:      IR Dump After DropAST

// CFGPASS: IR Dump Before FlattenCFG (cir-flatten-cfg)
