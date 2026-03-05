// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// XFAIL: *
//
// CIR crashes when handling virtual inheritance with thunks.
//
// Virtual inheritance requires:
// - VTT (Virtual Table Table) for construction
// - Virtual base pointer adjustments in thunks
// - vtable offset lookups for dynamic adjustment
//
// Currently, CIR crashes with:
//   Virtual adjustment NYI - requires vtable offset lookup
//   UNREACHABLE executed at CIRGenItaniumCXXABI.cpp:2203
//   at performTypeAdjustment during thunk generation
//
// This affects any class hierarchy using virtual inheritance.

struct Base {
    virtual ~Base() {}
    int b;
};

struct A : virtual Base {
    int a;
};

struct B : virtual Base {
    int b;
};

struct C : A, B {
    int c;
};

C* make_c() {
    return new C();
}

// LLVM: Should generate class with virtual inheritance
// LLVM: define {{.*}} @_Z6make_cv()

// OGCG: Should generate VTT and virtual base thunks
// OGCG: define {{.*}} @_Z6make_cv()
// OGCG: @_ZTT1C = {{.*}} VTT for C
