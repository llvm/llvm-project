// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

namespace GH69890 {
    // If the initializer is (), the object is value-initialized.
    struct A {
        constexpr A() {}
        int x;
    };

    struct B : A {
        int y;
    };

    static_assert(B().x == 0);
    static_assert(B().y == 0);
} // namespace GH69890

namespace P0960R3 {
    struct A { // expected-note 22 {{candidate constructor}}
        int i;
        operator int() volatile;
    };
    volatile A va;

    A a1(va);
    A a2 = va; // expected-error {{no matching constructor for initialization of 'A'}}
    A a3 {va};
    A a4 = {va}; // expected-error {{no matching constructor for initialization of 'A'}}

    A f() {
        return va; // expected-error {{no matching constructor for initialization of 'A'}}
        return {va}; // expected-error {{no matching constructor for initialization of 'A'}}
    }

    int g(A); // expected-note 2 {{passing argument to parameter here}}
    int i = g(va); // expected-error {{no matching constructor for initialization of 'A'}}
    int j = g({va}); // expected-error {{no matching constructor for initialization of 'A'}}

    struct Ambig {
        operator const A&(); // expected-note {{candidate function}}
        operator A&&(); // expected-note {{candidate function}}
        operator int();
    };

    A a5(Ambig {}); // expected-error {{call to constructor of 'A' is ambiguous}}
    A a6 = Ambig {}; // expected-error {{conversion from 'Ambig' to 'A' is ambiguous}}
    A a7 {Ambig {}};
    A a8 = {Ambig {}};

    A a9(1);
    A a10 = 1; // expected-error {{no viable conversion from 'int' to 'A'}}
    A a11 {1};
    A a12 = {1};


    struct B { // expected-note 12 {{candidate constructor}}
        int i;
        virtual operator int() volatile;
    };
    volatile B vb;

    B b1(vb); // expected-error {{no matching constructor for initialization of 'B'}}
    B b2 = vb; // expected-error {{no matching constructor for initialization of 'B'}}
    B b3 {vb}; // expected-error {{no matching constructor for initialization of 'B'}}
    B b4 = {vb}; // expected-error {{no matching constructor for initialization of 'B'}}


    struct Immovable {
        Immovable();
        Immovable(const Immovable&) = delete; // #Imm_copy
    };

    struct C { // #C
        int i;
        Immovable j; // #C_j

        operator int() volatile;
    };
    C c;
    volatile C vc;

    C c1(c); // expected-error {{call to implicitly-deleted copy constructor of 'C'}}
    C c2 = c; // expected-error {{call to implicitly-deleted copy constructor of 'C'}}
    C c3 {c}; // expected-error {{call to implicitly-deleted copy constructor of 'C'}}
    C c4 = {c}; // expected-error {{call to implicitly-deleted copy constructor of 'C'}}
    // expected-note@#C_j 4 {{copy constructor of 'C' is implicitly deleted}}
    // expected-note@#Imm_copy 4 {{'Immovable' has been explicitly marked deleted here}}

    C c5(vc);
    C c6 = vc; // expected-error {{no matching constructor for initialization of 'C'}}
    C c7 {vc};
    C c8 = {vc}; // expected-error {{no matching constructor for initialization of 'C'}}
    // expected-note@#C 4 {{candidate constructor}}

    C c9(C {});
    C c10 = C(123);
    C c11 {C {0, Immovable()}};
    C c12 = {C()};


    struct D { // expected-note 6 {{candidate constructor}}
        int i;
    };

    struct DD : private D { // expected-note 4 {{declared private here}}
        virtual operator int() volatile;
    };
    DD dd;
    volatile DD vdd;

    D d1(dd); // expected-error {{cannot cast 'const DD' to its private base class 'const D'}}
    D d2 = dd; // expected-error {{cannot cast 'const DD' to its private base class 'const D'}}
    D d3 {dd}; // expected-error {{cannot cast 'const DD' to its private base class 'const D'}}
    D d4 = {dd}; // expected-error {{cannot cast 'const DD' to its private base class 'const D'}}

    D d5(vdd);
    D d6 = vdd; // expected-error {{no matching constructor for initialization of 'D'}}
    D d7 {vdd};
    D d8 = {vdd}; // expected-error {{no matching constructor for initialization of 'D'}}
} // namespace P0960R3
