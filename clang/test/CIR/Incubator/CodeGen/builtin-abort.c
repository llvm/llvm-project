// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void abort();
void test() { abort(); }

// TODO: Add test to test unreachable when CIR support for NORETURN is added.

// CIR-LABEL: test
// CIR:  cir.call @abort() : () -> ()

// LLVM-LABEL: test
// LLVM:  call void @abort()
