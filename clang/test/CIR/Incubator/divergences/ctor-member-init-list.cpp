// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Constructor with member initializer list missing comdat.
//
// CodeGen:
//   $_ZN5OuterC1Ev = comdat any
//   $_ZN5InnerC1Ei = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZN5OuterC1Ev = comdat any

struct Inner {
    int val;
    Inner(int v) : val(v) {}
};

struct Outer {
    Inner i1, i2;
    Outer() : i1(10), i2(20) {}
};

int test() {
    Outer o;
    return o.i1.val + o.i2.val;
}
