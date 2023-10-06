// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM
// XFAIL: *

int *p = (int*)0x1234;


// CIR:  cir.global external @p = #cir.ptr<4660> : !cir.ptr<!s32i>
// LLVM: @p = global ptr inttoptr (i64 4660 to ptr)
