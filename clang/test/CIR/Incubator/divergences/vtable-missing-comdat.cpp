// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Vtable-related structures are missing comdat groups.
//
// CodeGen generates comdat for vtables and related structures:
//   $_ZTV4Base = comdat any
//   $_ZTI4Base = comdat any  (type info)
//   $_ZTS4Base = comdat any  (type string)
//
// CIR omits these comdat declarations:
//   @_ZTV4Base = linkonce_odr ... // No comdat
//
// This affects:
// - Vtables ($_ZTV*)
// - Type info structures ($_ZTI*)
// - Type name strings ($_ZTS*)
// - Virtual destructors
// - Virtual function overrides
//
// Impact: Linker cannot merge duplicates, potential code bloat

// DIFF: -$_ZTV4Base = comdat any
// DIFF: -$_ZTI4Base = comdat any
// DIFF: -$_ZTS4Base = comdat any
// DIFF: +# Missing comdat declarations

// Simple vtable
struct Base {
    virtual ~Base() {}
    virtual int get() { return 1; }
};

int test_simple_vtable() {
    Base b;
    return b.get();
}

// Virtual function override
struct Derived : Base {
    int get() override { return 2; }
};

int test_override() {
    Derived d;
    Base* b = &d;
    return b->get();
}

// Multiple virtual functions
struct Multi {
    virtual int foo() { return 1; }
    virtual int bar() { return 2; }
    virtual int baz() { return 3; }
    virtual ~Multi() {}
};

int test_multiple() {
    Multi m;
    return m.foo() + m.bar() + m.baz();
}

// Pure virtual
struct Abstract {
    virtual int get() = 0;
    virtual ~Abstract() {}
};

struct Concrete : Abstract {
    int get() override { return 42; }
};

int test_pure_virtual() {
    Concrete c;
    Abstract* a = &c;
    return a->get();
}

// Virtual with parameters
struct WithParams {
    virtual int add(int a, int b) { return a + b; }
    virtual ~WithParams() {}
};

struct WithParamsDerived : WithParams {
    int add(int a, int b) override { return a + b + 1; }
};

int test_params() {
    WithParamsDerived d;
    WithParams* p = &d;
    return p->add(10, 20);
}
