// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// dynamic_cast for upcasting missing type info comdat.
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
    int x = 10;
};

struct Derived : Base {
    int y = 20;
};

int test() {
    Derived d;
    Base* b = dynamic_cast<Base*>(&d);
    return b ? b->x : 0;
}
