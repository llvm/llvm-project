// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Parameterized constructor missing comdat.
//
// CodeGen:
//   $_ZN1SC1Eii = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZN1SC1Eii = comdat any

struct S {
    int x, y;
    S(int a, int b) : x(a), y(b) {}
};

int test() {
    S s(10, 20);
    return s.x + s.y;
}
