// RUN: %clang_cc1 -std=c++20 -fcxx-exceptions -verify %s

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

// A template parameter object shall have constant destruction.
namespace ConstDestruction {
  struct D {
    int n;
    bool can_destroy;

    constexpr ~D() {
      if (!can_destroy)
        throw "oh no"; // expected-note {{subexpression not valid}}
    }
  };

  template<D d>
  void f() {} // expected-note 2{{invalid explicitly-specified argument}}

  void g() {
    f<D{0, true}>();
    f<D{0, false}>(); // expected-error {{no matching function}}
  }

  // We can SFINAE on constant destruction.
  template<typename T> auto h(T t) -> decltype(f<T{1, false}>());
  template<typename T> auto h(T t) -> decltype(f<T{1, true}>());

  void i() {
    h(D());
    // Ensure we don't cache an invalid template argument after we've already
    // seen it in a SFINAE context.
    f<D{1, false}>(); // expected-error {{no matching function}}
    f<D{1, true}>();
  }

  template<D d> struct Z {};
  Z<D{2, true}> z1;
  Z<D{2, false}> z2; // expected-error {{non-type template argument is not a constant expression}} expected-note-re {{in call to '{{.*}}.~D()'}}
}
