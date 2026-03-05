// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// dynamic_cast for downcasting missing type info comdat.
//
// CodeGen:
//   $_ZTI4Base = comdat any
//   $_ZTI7Derived = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZTI4Base = comdat any
// DIFF: -$_ZTI7Derived = comdat any

struct Base {
    virtual ~Base() {}
};

struct Derived : Base {
    int value = 42;
};

int test() {
    Derived d;
    Base* b = &d;
    Derived* dp = dynamic_cast<Derived*>(b);
    return dp ? dp->value : 0;
}
