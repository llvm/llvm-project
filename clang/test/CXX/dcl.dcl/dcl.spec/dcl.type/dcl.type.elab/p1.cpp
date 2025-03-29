// RUN: %clang_cc1 -verify %s -std=c++11

namespace N {
  struct A;
  template<typename T> struct B {};
}
template<typename T> struct C {};
struct D {
  template<typename T> struct A {};
};
struct N::A; // expected-error {{cannot have a nested name specifier}}

template<typename T> struct N::B; // expected-error {{cannot have a nested name specifier}}
template<typename T> struct N::B<T*>; // FIXME: This is technically ill-formed, but that's not the intent.
template<> struct N::B<int>;
template struct N::B<float>;

template<typename T> struct C;
template<typename T> struct C<T*>;
template<> struct C<int>;
template struct C<float>;

template<typename T> struct D::A; // expected-error {{cannot have a nested name specifier}}
template<typename T> struct D::A<T*>; // FIXME: This is technically ill-formed, but that's not the intent.
template<> struct D::A<int>;
template struct D::A<float>;

namespace qualified_decl {
  template<typename T>
  struct S0 {
    struct S1;

    template<typename U>
    struct S2;

    enum E0 : int;

    enum class E1;
  };

  struct S3 {
    struct S4;

    template<typename T>
    struct S5;

    enum E2 : int;

    enum class E3;
  };

  template<typename T>
  struct S0<T>::S1; // expected-error{{cannot have a nested name specifier}}

  template<>
  struct S0<int>::S1;

  template<typename T>
  template<typename U>
  struct S0<T>::S2; // expected-error{{cannot have a nested name specifier}}

  template<typename T>
  template<typename U>
  struct S0<T>::S2<U*>;

  template<>
  template<>
  struct S0<int>::S2<bool>;

  template<>
  template<typename U>
  struct S0<int>::S2;

  struct S3::S4; // expected-error{{cannot have a nested name specifier}}

  template<typename T>
  struct S3::S5; // expected-error{{cannot have a nested name specifier}}

  struct S3::S4 f0();
  enum S0<long>::E0 f1();
  enum S0<long>::E1 f2();
  enum S3::E2 f3();
  enum S3::E3 f4();

  using A0 = struct S3::S4;
  using A1 = enum S0<long>::E0;
  using A2 = enum S0<long>::E1;
  using A3 = enum S3::E2;
  using A4 = enum S3::E3;
}
