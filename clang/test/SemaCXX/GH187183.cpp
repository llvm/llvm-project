// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace GH187183 {
    template<int T>
    struct S {
        S();
    };

    using X = S<0>;

    template<int T> // expected-error {{template parameter list matching the non-templated nested type 'GH187183::S<0>' should be empty ('template<>')}}
    S<0>::S() {
        int e[1];
        test([&e]() { return e[0]; });
    }
}
