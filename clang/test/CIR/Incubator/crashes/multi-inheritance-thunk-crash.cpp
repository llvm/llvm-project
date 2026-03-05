// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// XFAIL: *
//
// CIR crashes when generating thunks for multiple inheritance.
//
// Multiple inheritance requires generating thunks to adjust the 'this' pointer
// when calling virtual functions through a base class pointer that is not the
// primary base. The thunk adjusts 'this' and then calls the actual implementation.
//
// Currently, CIR crashes with:
//   Assertion `isValid()' failed in Address::getPointer()
//   at clang::CIRGen::CIRGenFunction::emitReturnOfRValue
//
// This affects any class using multiple inheritance with virtual functions.

struct A {
    virtual ~A() {}
    virtual int foo() { return 1; }
    int a;
};

struct B {
    virtual ~B() {}
    virtual int bar() { return 2; }
    int b;
};

struct C : A, B {
    int foo() override { return 3; }
    int bar() override { return 4; }
};

C* make_c() {
    return new C();
}

// LLVM: Should generate thunks for B's vtable in C
// LLVM: define {{.*}} @_Z6make_cv()

// OGCG: Should generate thunks for B's vtable in C
// OGCG: define {{.*}} @_Z6make_cv()
// OGCG: define {{.*}} @_ZThn{{[0-9]+}}_N1C3barEv
