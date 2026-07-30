// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s

template <typename T> struct A {
  A(void *) {}
  T(A<T>){}; // expected-error{{member 'A' cannot have template arguments}}\
  // expected-error2{{member 'A' has the same name as its class}}
};
// Don't crash.
A<int> instantiate1() { return {nullptr}; } // expected-note{{in instantiation of template class 'A<int>' requested here}}

template <typename T> struct B {
  B(void *) {}
  T B<T>{}; // expected-error{{member 'B' cannot have template arguments}}\
  // expected-error2{{member 'B' has the same name as its class}}
};
// Don't crash.
B<int> instantiate2() { return {nullptr}; } // expected-note{{in instantiation of template class 'B<int>' requested here}}

template <typename T> struct S {};

template <typename T> struct C {
  C(void *) {}
  T S<T>{}; // expected-error{{member 'S' cannot have template arguments}}
};
// Don't crash.
C<int> instantiate3() { return {nullptr}; }

template <typename T, template <typename> typename U> class D {
public:
  D(void *) {}
  T(D<T, U<T>>) {} // expected-error{{member 'D' cannot have template arguments}}\
  // expected-error{{expected ';' at end of declaration list}}\
  // expected-error2{{member 'D' has the same name as its class}}
};
// Don't crash.
D<int, S> instantiate4() { return D<int, S>(nullptr); } // expected-note{{in instantiation of template class 'D<int, S>' requested here}}
