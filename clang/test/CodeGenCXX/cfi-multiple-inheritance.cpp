// Test that correct vtable ptr and type metadata are passed to llvm.type.test
// Related to Bugzilla 43390.

// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux -fvisibility hidden -std=c++11 -fsanitize=cfi-nvcall -emit-llvm -o - %s | FileCheck %s

class A1 {
public:
    virtual int f1() = 0;
};

class A2 {
public:
    virtual int f2() = 0;
};


class B : public A1, public A2 {
public:
    int f2() final { return 1; }
    int f1() final { return 2; }
};

// CHECK-LABEL: define hidden noundef i32 @_Z3foov
int foo() {
    B b;
    return static_cast<A2*>(&b)->f2();
    // CHECK: [[P:%[^ ]*]] = bitcast %class.B* %b to i8**
    // CHECK: [[V:%[^ ]*]] = load i8*, i8** [[P]], align 8
    // CHECK: call i1 @llvm.type.test(i8* [[V]], metadata !"_ZTS1B")
    // CHECK: call i1 @llvm.type.test(i8* [[V]], metadata !"all-vtables")
}
