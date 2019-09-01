// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s -Wfinal-dtor-non-final-class

class A {
    ~A();
};

class B { // expected-note {{mark 'B' as 'final' to silence this warning}}
    virtual ~B() final; // expected-warning {{class with destructor marked 'final' cannot be inherited from}}
};

class C final {
    virtual ~C() final;
};
