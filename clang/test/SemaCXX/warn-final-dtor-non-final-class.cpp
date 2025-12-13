// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s -Wfinal-dtor-non-final-class
// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -Wfinal-dtor-non-final-class -fdiagnostics-parseable-fixits 2>&1 | FileCheck %s

class A {
    ~A();
};

class B { // expected-note {{mark 'B' as 'final' to silence this warning}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:8-[[@LINE-1]]:8}:" final"
    virtual ~B() final; // expected-warning {{class with destructor marked 'final' cannot be inherited from}}
};

class C final {
    virtual ~C() final;
};
