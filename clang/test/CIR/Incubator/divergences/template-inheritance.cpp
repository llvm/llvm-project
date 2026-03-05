// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Template inheritance missing comdat.
//
// CodeGen:
//   $_ZN7DerivedC1Ei = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZN7DerivedC1Ei = comdat any

template<typename T>
struct Base {
    T value;
    Base(T v) : value(v) {}
};

struct Derived : Base<int> {
    Derived(int v) : Base<int>(v) {}
};

int test() {
    Derived d(42);
    return d.value;
}
