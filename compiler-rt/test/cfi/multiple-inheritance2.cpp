// Test that virtual functions of the derived class can be called through
// pointers of both base classes without CFI errors.
// Related to Bugzilla 43390.

// RUN: %clangxx_cfi -o %t1 %s
// RUN: %run %t1 2>&1 | FileCheck --check-prefix=CFI %s

// CFI: In f1
// CFI: In f2
// CFI-NOT: control flow integrity check

// REQUIRES: cxxabi

#include <stdio.h>

class A1 {
public:
    virtual void f1() = 0;
};

class A2 {
public:
    virtual void f2() = 0;
};


class B : public A1, public A2 {
public:
    void f2() final { fprintf(stderr, "In f2\n"); }
    void f1() final { fprintf(stderr, "In f1\n"); }
};

int main() {
    B b;

    static_cast<A1*>(&b)->f1();
    static_cast<A2*>(&b)->f2();
}
