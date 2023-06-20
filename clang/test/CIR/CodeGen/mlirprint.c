// RUN: %clang_cc1 -fclangir -emit-cir -mmlir --mlir-print-ir-after-all %s -o %t.cir 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fclangir -emit-cir -mmlir --mlir-print-ir-after-all %s -o %t.ll 2>&1 | FileCheck %s

int foo() {
  int i = 3;
  return i;
}


// CHECK: IR Dump After MergeCleanups (cir-merge-cleanups)
// cir.func @foo() -> !s32i
// CHECK: IR Dump After DropAST (cir-drop-ast)
// cir.func @foo() -> !s32i
