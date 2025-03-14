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
    struct A { // expected-note 9 {{candidate constructor}}
        int i;
        operator int() volatile;
    };

    volatile A va;
    A a = va; // expected-error {{no matching constructor for initialization of 'A'}}

    A f() {
        return va; // expected-error {{no matching constructor for initialization of 'A'}}
    }

    int g(A); // expected-note {{passing argument to parameter here}}
    int g(auto&&);
    int i = g(va); // expected-error {{no matching constructor for initialization of 'A'}}
} // namespace P0960R3
