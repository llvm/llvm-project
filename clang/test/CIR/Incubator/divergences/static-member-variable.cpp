// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Static member variable.
//
// CodeGen:
//   @_ZN1S7counterE = global i32 100
//
// CIR:
//   Check for differences

// DIFF: Check for static member handling

struct S {
    static int counter;
};

int S::counter = 100;

int test() {
    return S::counter;
}
