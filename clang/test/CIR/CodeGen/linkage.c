// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM


static int bar(int i) {
  return i;
}

int foo() {
  return bar(5);
}

// CIR:   cir.func internal private @bar(
// CIR:   cir.func @foo(

// LLVM: define internal i32 @bar(
// LLVM: define i32 @foo(
