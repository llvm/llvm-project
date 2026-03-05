// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Private inheritance constructor missing comdat.
//
// CodeGen:
//   $_ZN7DerivedC1Ev = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZN7DerivedC1Ev = comdat any

struct Base {
    int x = 42;
};

struct Derived : private Base {
    int get_x() { return x; }
};

int test() {
    Derived d;
    return d.get_x();
}
