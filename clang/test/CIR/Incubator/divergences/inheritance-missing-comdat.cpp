// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Inheritance-related functions are missing comdat groups.
//
// CodeGen generates comdat for:
// - Constructors of derived classes
// - Destructors of derived classes
// - Base-to-derived conversions
// - Virtual function overrides in derived classes
//
// CIR omits these comdat declarations.
//
// Impact: ODR violations possible with multiple inheritance hierarchies

// DIFF: -$_ZN7DerivedC1Ev = comdat any
// DIFF: -$_ZN7DerivedD1Ev = comdat any
// DIFF: +# Missing comdat declarations

// Simple single inheritance
struct Base {
    int x = 10;
    virtual ~Base() {}
};

struct Derived : Base {
    int y = 20;
    ~Derived() override {}
};

int test_single_inheritance() {
    Derived d;
    return d.x + d.y;
}

// Multiple inheritance
struct A {
    int a = 1;
    virtual ~A() {}
};

struct B {
    int b = 2;
    virtual ~B() {}
};

struct C : A, B {
    int c = 3;
    ~C() override {}
};

int test_multiple_inheritance() {
    C obj;
    return obj.a + obj.b + obj.c;
}

// Three levels of inheritance
struct GrandParent {
    int gp = 1;
    virtual ~GrandParent() {}
};

struct Parent : GrandParent {
    int p = 2;
    ~Parent() override {}
};

struct Child : Parent {
    int c = 3;
    ~Child() override {}
};

int test_deep_inheritance() {
    Child ch;
    return ch.gp + ch.p + ch.c;
}

// Constructor inheritance
struct BaseWithCtor {
    int x;
    BaseWithCtor(int val) : x(val) {}
    virtual ~BaseWithCtor() {}
};

struct DerivedWithCtor : BaseWithCtor {
    using BaseWithCtor::BaseWithCtor;
    ~DerivedWithCtor() override {}
};

int test_ctor_inheritance() {
    DerivedWithCtor d(42);
    return d.x;
}

// Empty base optimization
struct Empty {};

struct DerivedFromEmpty : Empty {
    int x = 42;
};

int test_empty_base() {
    DerivedFromEmpty d;
    return d.x;
}
