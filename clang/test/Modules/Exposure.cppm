// RUN: %clang_cc1 -std=c++20 %s -verify -fsyntax-only
export module M;
namespace {
class TULocalClass {};
}

template <typename T>
class Templ {};

class C {
    TULocalClass foo() { return TULocalClass(); } // expected-warning {{TU local entity 'TULocalClass' is exposed}}
private:
    TULocalClass Member; // expected-warning {{TU local entity 'TULocalClass' is exposed}}
};

static inline void its() {}
template<int> void g() { its(); }

void f0() {
    g<1>();
}

inline void f1() {
    g<1>(); // expected-warning {{TU local entity 'g<1>' is exposed}}
}
