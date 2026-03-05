// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Deep inheritance chain constructors missing comdat.
//
// CodeGen:
//   $_ZN5ChildC1Ev = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZN5ChildC1Ev = comdat any

struct GrandParent {
    int a = 1;
};

struct Parent : GrandParent {
    int b = 2;
};

struct Child : Parent {
    int c = 3;
};

int test() {
    Child ch;
    return ch.a + ch.b + ch.c;
}
