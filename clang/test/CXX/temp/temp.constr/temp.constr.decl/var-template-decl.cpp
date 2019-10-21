// RUN: %clang_cc1 -std=c++2a -x c++ -verify %s

namespace nodiag {

struct B {
    template <typename T> requires bool(T())
    static int A;
};

template <typename U> requires bool(U())
int B::A = int(U());

} // end namespace nodiag

namespace diag {

struct B {
    template <typename T> requires bool(T()) // expected-note{{previous template declaration is here}}
    static int A;
};

template <typename U> requires !bool(U())  // expected-error{{requires clause differs in template redeclaration}}
int B::A = int(U());

} // end namespace diag