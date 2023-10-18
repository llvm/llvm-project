// RUN: %clang_cc1 -std=c++17 -verify %s
// expected-no-diagnostics

template<class T> struct S {
    template<class U> struct N {
        N(T) {}
        N(T, U) {}
        template<class V> N(V, U) {}
    };
};

S<int>::N x{"a", 1};
