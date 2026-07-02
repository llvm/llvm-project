// RUN: %clang_cc1 -verify -std=c++23 %s -fsyntax-only

struct S {}; // expected-note 2{{candidate constructor}}
template <typename T> struct SS { T t1; T t2; };
template <class T, class... Args> T C(Args... args) { return SS("foo"); } // expected-error {{no viable conversion}}
S s = C<S>(); // expected-note {{in instantiation of function template specialization 'C<S>'}}

template <class T> struct SS2 { T t1, t2; };
template <class> void C2() {
  SS2("foo"); // expected-warning 2{{expression result unused}}
}
template void C2<int>(); // expected-note {{in instantiation of function template specialization 'C2<int>'}}
