// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Pointer to virtual member function.
// Virtual member pointers encode vtable offset and need special handling.
//
// CodeGen decomposes:
//   define i32 @call_virtual_ptr(ptr %b, i64 %ptr.coerce0, i64 %ptr.coerce1)
//
// CIR passes as struct:
//   define i32 @call_virtual_ptr(ptr %0, { i64, i64 } %1)

// DIFF: -define {{.*}} @{{.*}}call_virtual_ptr(ptr{{.*}}, i64{{.*}}, i64
// DIFF: +define {{.*}} @{{.*}}call_virtual_ptr(ptr{{.*}}, { i64, i64 }

struct Base {
    virtual int foo() { return 1; }
};

struct Derived : Base {
    int foo() override { return 2; }
};

int call_virtual_ptr(Base* b, int (Base::*ptr)()) {
    return (b->*ptr)();
}

int test() {
    Derived d;
    return call_virtual_ptr(&d, &Base::foo);
}
