// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Multiple inheritance constructors missing comdat.
//
// CodeGen:
//   $_ZN1CC1Ev = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZN1CC1Ev = comdat any

struct A {
    int a = 1;
};

struct B {
    int b = 2;
};

struct C : A, B {
    int c = 3;
};

int test() {
    C obj;
    return obj.a + obj.b + obj.c;
}
