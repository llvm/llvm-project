// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Constructor inheritance missing comdat.
//
// CodeGen:
//   $_ZN7DerivedC1Ei = comdat any (inherited)
//
// CIR:
//   Missing comdat

// DIFF: -$_ZN7DerivedC1Ei = comdat any

struct Base {
    int x;
    Base(int val) : x(val) {}
};

struct Derived : Base {
    using Base::Base;  // Inherit constructors
};

int test() {
    Derived d(42);
    return d.x;
}
