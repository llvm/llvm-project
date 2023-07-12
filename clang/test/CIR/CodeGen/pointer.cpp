// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// FIXME(cir): Move the test below to lowering and us a separate tool to lower from CIR to LLVM IR.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

// Global pointer should be zero initialized by default.
int *ptr;
// CHECK: cir.global external @ptr = #cir.null : !cir.ptr<!s32i>
// LLVM: @ptr = global ptr null
