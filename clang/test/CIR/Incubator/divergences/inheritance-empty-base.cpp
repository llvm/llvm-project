// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Empty base optimization.
//
// CodeGen:
//   $_ZN7DerivedC1Ev = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZN7DerivedC1Ev = comdat any

struct Empty {};

struct Derived : Empty {
    int x = 42;
};

int test() {
    Derived d;
    return d.x;
}
