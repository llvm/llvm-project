// RUN: %clang_cc1 -std=c++20 -fclangir -emit-cir -triple x86_64-unknown-linux-gnu %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR

class A {
public:
    A(void *, ...);
};

A a(nullptr, 1, "str");

// CIR: cir.func {{.*}} @_ZN1AC1EPvz(!cir.ptr<!rec_A>, !cir.ptr<!void>, ...)
