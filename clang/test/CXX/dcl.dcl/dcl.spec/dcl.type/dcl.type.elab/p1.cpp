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
template<typename T> struct C<T*>; // FIXME: This is technically ill-formed, but that's not the intent.
template<> struct C<int>;
template struct C<float>;

template<typename T> struct D::A; // expected-error {{cannot have a nested name specifier}}
template<typename T> struct D::A<T*>; // FIXME: This is technically ill-formed, but that's not the intent.
template<> struct D::A<int>;
template struct D::A<float>;
