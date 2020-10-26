// RUN: %clang_cc1 -std=c++20 -verify %s

struct A { int n; };

template<A a> struct B {
  static constexpr A &v = a; // expected-error {{binding reference of type 'A' to value of type 'const A' drops 'const' qualifier}}
};

template<A a> struct C {
  static constexpr const A &v = a;
};

// All such template parameters in the program of the same type with the same
// value denote the same template parameter object.
template<A a, typename T> void check() {
  static_assert(&a == &T::v); // expected-error {{failed}}
}

using T = C<A{1}>;
template void check<A{1}, T>();
template void check<A{2}, T>(); // expected-note {{instantiation of}}

// Different types with the same value are unequal.
struct A2 { int n; };
template<A2 a2> struct C2 {
  static constexpr const A2 &v = a2;
};
static_assert((void*)&C<A{}>::v != (void*)&C2<A2{}>::v);
