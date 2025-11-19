// RUN: %clang_cc1 -std=c++20 %s -fsyntax-only -verify
export module m;

namespace test {
namespace ns1 {
    namespace ns2 {
    template<class T> void f(T t); // expected-note {{target of using declaration}}
    }
    using ns2::f; // expected-note {{using declaration}}
}
struct A { void f(); }; // expected-note 2{{target of using declaration}}
struct B : public A { using A::f; }; // expected-note {{using declaration}}
template<typename T> struct C : A { using A::f; }; // expected-note {{using declaration}}
struct X {
    template<class T> friend void ns1::f(T t); // expected-error {{cannot befriend target of using declaration}}
    friend void B::f(); // expected-error {{cannot befriend target of using declaration}}
    friend void C<int>::f(); // expected-error {{cannot befriend target of using declaration}}
};
}
