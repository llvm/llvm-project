// RUN: %clang_cc1 -std=c++14 -Wno-unused-value -verify %s

// A glvalue of type "cv1 T1" can be cast to type "rvalue reference to
// cv2 T2" if "cv2 T2" is reference-compatible with "cv1 T1" (8.5.3).
struct A { };
struct B : A { };

template<typename T> T& lvalue();
template<typename T> T&& xvalue();

void test(A &a, B &b) {
  A &&ar0 = static_cast<A&&>(a);
  A &&ar1 = static_cast<A&&>(b);
  A &&ar2 = static_cast<A&&>(lvalue<A>());
  A &&ar3 = static_cast<A&&>(lvalue<B>());
  A &&ar4 = static_cast<A&&>(xvalue<A>());
  A &&ar5 = static_cast<A&&>(xvalue<B>());
  const A &&ar6 = static_cast<const A&&>(a);
  const A &&ar7 = static_cast<const A&&>(b);
  const A &&ar8 = static_cast<const A&&>(lvalue<A>());
  const A &&ar9 = static_cast<const A&&>(lvalue<B>());
  const A &&ar10 = static_cast<const A&&>(xvalue<A>());
  const A &&ar11 = static_cast<const A&&>(xvalue<B>());
}

namespace GH121429 {

struct C : private A { // expected-note 4 {{declared private here}}
    C&& that();

    void f() {
        static_cast<A&&>(*this);
        static_cast<const A&&>(*this);

        static_cast<A&&>(that());
        static_cast<const A&&>(that());
    }
};
C c;
const C cc;

void f() {
    static_cast<A&&>(c);        // expected-error {{cannot cast 'C' to its private base class 'A'}}
    static_cast<A&&>(c.that()); // expected-error {{cannot cast 'C' to its private base class 'A'}}

    static_cast<const A&&>(c);        // expected-error {{cannot cast 'C' to its private base class 'const A'}}
    static_cast<const A&&>(c.that()); // expected-error {{cannot cast 'C' to its private base class 'const A'}}
}

constexpr bool g() {
    (A&&)c;
    (A&&)(C&&)c;
    (A&&)cc;
    (A&&)(const C&&)c;
    (const A&&)c;
    (const A&&)(C&&)c;
    (const A&&)cc;
    (const A&&)(const C&&)c;
    return true;
}
static_assert(g(), "");

struct D : A, B { // expected-warning {{direct base 'A' is inaccessible due to ambiguity}}
    D&& rv();
};
D d;

void h(const D cd) {
    static_cast<A&&>(d);      // expected-error {{ambiguous conversion from derived class 'D' to base class 'A'}}
    static_cast<A&&>(d.rv()); // expected-error {{ambiguous conversion from derived class 'D' to base class 'A'}}

    static_cast<const A&&>(d);      // expected-error {{ambiguous conversion from derived class 'D' to base class 'const A'}}
    static_cast<const A&&>(d.rv()); // expected-error {{ambiguous conversion from derived class 'D' to base class 'const A'}}

    (A&&)d;                  // expected-error {{ambiguous conversion from derived class 'D' to base class 'A'}}
    (A&&)(D&&)d;             // expected-error {{ambiguous conversion from derived class 'D' to base class 'A'}}
    (A&&)cd;                 // expected-error {{ambiguous conversion from derived class 'D' to base class 'A'}}
    (A&&)(const D&&)d;       // expected-error {{ambiguous conversion from derived class 'D' to base class 'A'}}
    (const A&&)d;            // expected-error {{ambiguous conversion from derived class 'D' to base class 'A'}}
    (const A&&)(D&&)d;       // expected-error {{ambiguous conversion from derived class 'D' to base class 'A'}}
    (const A&&)cd;           // expected-error {{ambiguous conversion from derived class 'D' to base class 'A'}}
    (const A&&)(const D&&)d; // expected-error {{ambiguous conversion from derived class 'D' to base class 'A'}}
}

template<class T, class U>
auto s(U u = {}) -> decltype(static_cast<T&&>(u)); // expected-note 2 {{substitution failure}}

int i = s<A, C>(); // expected-error {{no matching function}}
int j = s<A, D>(); // expected-error {{no matching function}}

}
