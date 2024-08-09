// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm-bc %s -o %t.bc
// RUN: llvm-dis %t.bc -o %t.bc.ll
// RUN: FileCheck --input-file=%t.bc.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -S %s -o %t.s
// RUN: FileCheck --input-file=%t.s %s -check-prefix=ASM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-obj %s -o %t.o
// RUN: llvm-objdump -d %t.o | FileCheck %s -check-prefix=OBJ

void foo() {}

//      MLIR: func.func @foo() {
// MLIR-NEXT:   return
// MLIR-NEXT: }

//      LLVM: define dso_local void @foo()
// LLVM-NEXT:   ret void
// LLVM-NEXT: }

//      ASM: .globl  foo
// ASM-NEXT: .p2align
// ASM-NEXT: .type foo,@function
// ASM-NEXT: foo:
//      ASM: retq

// OBJ: 0: c3 retq
