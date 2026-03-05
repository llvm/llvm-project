// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Copy constructor missing comdat.
//
// CodeGen:
//   $_ZN1SC1ERKS_ = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZN1SC1ERKS_ = comdat any

struct S {
    int x;
    S(int val) : x(val) {}
    S(const S& other) : x(other.x * 2) {}
};

int test() {
    S s1(10);
    S s2(s1);
    return s2.x;
}
